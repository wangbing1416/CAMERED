import os
import json
import random
import logging
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from torch.cuda.amp import autocast, GradScaler

import multiprocessing


def set_spawn_start_method():
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyDataset(Dataset):
    def __init__(self, data_file: str, tokenizer: BertTokenizer, max_length: int = 128):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.data.append(item)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[int, str, List[str]]:
        item = self.data[idx]
        label = item['label']
        tweet = item.get('tweet') or item.get('content', '')
        comments = item.get('comments', []).copy()
        random.shuffle(comments)
        return label, tweet, comments


def collate_fn(batch: List[Tuple[int, str, List[str]]], tokenizer: BertTokenizer, max_length: int):
    labels, tweets, comments = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)

    encoded_tweets = tokenizer(
        list(tweets),
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=max_length,
        return_token_type_ids=True
    )

    all_comments = [comment for comments_per_sample in comments for comment in comments_per_sample]
    if all_comments:
        encoded_comments = tokenizer(
            all_comments,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=max_length,
            return_token_type_ids=True
        )
    else:
        # if no comment, create blank tensor
        encoded_comments = {
            'input_ids': torch.empty((0, max_length), dtype=torch.long),
            'attention_mask': torch.empty((0, max_length), dtype=torch.long),
            'token_type_ids': torch.empty((0, max_length), dtype=torch.long)
        }

    return labels, encoded_tweets, comments, encoded_comments


class CollateFnWrapper:
    def __init__(self, tokenizer: BertTokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Tuple[int, str, List[str]]]):
        return collate_fn(batch, self.tokenizer, self.max_length)


class MyModel(nn.Module):
    def __init__(self, bert_model_path: str, t: float, num_labels: int = 4, hidden_size: int = 768,
                 max_length: int = 128):
        super(MyModel, self).__init__()
        self.t = t
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.bert = BertModel.from_pretrained(bert_model_path)

        self.attention_plus = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)
        self.attention_minus = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)

        self.W = nn.Linear(4 * hidden_size, hidden_size)

        intermediate_size = hidden_size // 2

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, intermediate_size),
            nn.BatchNorm1d(intermediate_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(intermediate_size, self.num_labels)
        )

    def forward(self, encoded_tweets: dict, encoded_comments: dict, comments_list: List[List[str]]) -> torch.Tensor:
        """
        前向传播方法，输出维度为 [batch_size, num_labels]
        """
        batch_size = encoded_tweets['input_ids'].size(0)

        with torch.no_grad():
            tweet_outputs = self.bert(
                input_ids=encoded_tweets['input_ids'],
                attention_mask=encoded_tweets['attention_mask'],
                token_type_ids=encoded_tweets['token_type_ids']
            )
        tweet_embeddings = tweet_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        comment_embeddings = None
        if encoded_comments['input_ids'].shape[0] > 0:
            with torch.no_grad():
                comment_outputs = self.bert(
                    input_ids=encoded_comments['input_ids'],
                    attention_mask=encoded_comments['attention_mask'],
                    token_type_ids=encoded_comments['token_type_ids']
                )
            comment_embeddings = comment_outputs.last_hidden_state[:, 0, :]  # [total_comments, hidden_size]
        else:
            comment_embeddings = torch.empty((0, self.hidden_size), device=device)

        hc_batch = []
        idx = 0

        for i in range(batch_size):
            tweet_embedding = tweet_embeddings[i]  # [hidden_size]

            comments = comments_list[i]
            num_comments = len(comments)

            if num_comments == 0:
                hc = torch.zeros(self.hidden_size, device=tweet_embedding.device)
                combined = torch.cat([hc, tweet_embedding], dim=0)  # [2 * hidden_size]
                hc_batch.append(combined)
                continue

            current_comment_embeddings = comment_embeddings[idx:idx + num_comments]
            idx += num_comments

            tweet_emb_norm = F.normalize(tweet_embedding, dim=0)
            comment_emb_norm = F.normalize(current_comment_embeddings, dim=1)
            similarities = torch.matmul(comment_emb_norm, tweet_emb_norm)

            mask_plus = similarities > self.t
            mask_minus = similarities <= self.t

            embeddings_plus = current_comment_embeddings[mask_plus]
            embeddings_minus = current_comment_embeddings[mask_minus]

            if embeddings_plus.shape[0] == 0:
                h_plus = torch.zeros(self.hidden_size, device=tweet_embedding.device)
            else:
                with autocast():
                    attn_output_plus, _ = self.attention_plus(
                        embeddings_plus.unsqueeze(1), embeddings_plus.unsqueeze(1), embeddings_plus.unsqueeze(1)
                    )
                h_plus = attn_output_plus.mean(dim=0).squeeze(0)

            if embeddings_minus.shape[0] == 0:
                h_minus = torch.zeros(self.hidden_size, device=tweet_embedding.device)
            else:
                with autocast():
                    attn_output_minus, _ = self.attention_minus(
                        embeddings_minus.unsqueeze(1), embeddings_minus.unsqueeze(1), embeddings_minus.unsqueeze(1)
                    )
                h_minus = attn_output_minus.mean(dim=0).squeeze(0)

            h_concat = torch.cat([
                h_plus,
                h_plus * h_minus,
                h_plus - h_minus,
                h_minus
            ], dim=0)

            hc = self.W(h_concat)
            combined = torch.cat([hc, tweet_embedding], dim=0)
            hc_batch.append(combined)

        hc_batch = torch.stack(hc_batch, dim=0)
        output = self.classifier(hc_batch)  # [batch_size, num_labels]

        return output

def save_model(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, save_path: str):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(state, save_path)
    print(f"model has been saved in {save_path}")


def load_model(model: nn.Module, optimizer: torch.optim.Optimizer, load_path: str) -> Tuple[
    nn.Module, torch.optim.Optimizer, int]:
    if os.path.exists(load_path):
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"load model from {load_path}, trained from the {start_epoch} epoch")
        return model, optimizer, start_epoch
    else:
        print(f"not find {load_path}")
        return model, optimizer, 0


def evaluate_model(model: nn.Module, dataloader: DataLoader) -> float:
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            labels, encoded_tweets, comments, encoded_comments = batch
            labels = labels.to(device)
            for k in encoded_tweets:
                encoded_tweets[k] = encoded_tweets[k].to(device)
            for k in encoded_comments:
                if encoded_comments[k].numel() > 0:
                    encoded_comments[k] = encoded_comments[k].to(device)

            with autocast():
                outputs = model(encoded_tweets, encoded_comments, comments)  # [batch_size, num_labels]

            _, preds = torch.max(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"accuracy: {accuracy * 100:.2f}%")
    return accuracy


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler("best_accuracies.txt"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


def main(
        train_json: str,
        bert_model_path: str,
        model_saved_path: str,
        t: float,
        epochs: int,
        patience: int,
        val_json: str,
        test_json: str,
        num_labels:int,
        batch_size: int
):
    logger = setup_logger()
    logger.info(f"start training, t={t}")
    batch_size = 2
    num_epochs = epochs
    best_accuracy = 0
    best_epoch = 0
    no_improvement_count = 0

    tokenizer = BertTokenizer.from_pretrained(bert_model_path)

    # train and evaluation
    train_dataset = MyDataset(train_json, tokenizer, max_length=64)
    collate_wrapper = CollateFnWrapper(tokenizer, max_length=64)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_wrapper,
        drop_last=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_dataset = MyDataset(val_json, tokenizer, max_length=64)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_wrapper,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    model = MyModel(bert_model_path, t, num_labels).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {'params': model.attention_plus.parameters(), 'lr': 1e-5},
        {'params': model.attention_minus.parameters(), 'lr': 1e-5},
        {'params': model.W.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-5}
    ])

    num_training_steps = num_epochs * len(train_dataloader)
    num_warmup_steps = int(0.02 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            labels, encoded_tweets, comments, encoded_comments = batch

            labels = labels.to(device)
            for k in encoded_tweets:
                encoded_tweets[k] = encoded_tweets[k].to(device)
            for k in encoded_comments:
                if encoded_comments[k].numel() > 0:
                    encoded_comments[k] = encoded_comments[k].to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(encoded_tweets, encoded_comments, comments)  # [batch_size, num_labels]
                loss = criterion(outputs, labels)
            print(f"Epoch {epoch + 1}, Batch {step + 1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

            torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1}, average loss: {avg_loss:.4f}")

        accuracy = evaluate_model(model, val_dataloader)
        logger.info(f"Epoch {epoch + 1}, validate accuracy: {accuracy * 100:.2f}%")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch + 1
            no_improvement_count = 0
            save_model(model, optimizer, epoch, model_saved_path)
            logger.info(f"save the best model in {model_saved_path} (Epoch {epoch + 1})")
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            logger.info(f"Early stop, the best epoch: {best_epoch}, best accuracy: {best_accuracy * 100:.2f}%")
            break

    logger.info(f"training finish. The best Epoch: {best_epoch}, the best accuracy: {best_accuracy * 100:.2f}%\n")

    del train_dataset
    del train_dataloader
    del val_dataset
    del val_dataloader
    del model
    torch.cuda.empty_cache()

    # testing
    model = MyModel(bert_model_path, t, num_labels=4).to(device)
    optimizer = torch.optim.Adam([
        {'params': model.attention_plus.parameters(), 'lr': 1e-5},
        {'params': model.attention_minus.parameters(), 'lr': 1e-5},
        {'params': model.W.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-5}
    ])

    checkpoint = torch.load(model_saved_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()

    test_dataset = MyDataset(test_json, tokenizer, max_length=128)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_wrapper,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_accuracy = evaluate_model(model, test_dataloader)
    print(f"testing accuracy: {test_accuracy * 100:.2f}%")

    with open("best_accuracies.txt", "a") as f:
        f.write(f"Best Epoch: {best_epoch}, Best Accuracy: {best_accuracy * 100:.2f}%\n")

    del test_dataset
    del test_dataloader
    del model
    torch.cuda.empty_cache()



if __name__ == "__main__":
    import argparse
    set_spawn_start_method()

    parser = argparse.ArgumentParser(description="Train generator"
                                     )
    parser.add_argument('--train_json', type=str, required=False,
                        default="../data/twitter16/train.json",
                        help='Path to the first input JSON file')
    parser.add_argument('--val_json', type=str, required=False,
                        default="../data/twitter16/val.json",
                        help='Path to the first input JSON file')
    parser.add_argument('--test_json', type=str, required=False,
                        default="../data/twitter16/test.json",
                        help='Path to the first input JSON file')

    parser.add_argument('--bert_path', type=str, required=False, default="../bert/",
                        help='Path to the second input JSON file')
    parser.add_argument('--model_saved_path', type=str, required=False,
                        default="../data/twitter16/final_result/",
                        help='Path to pretrained model or model name')
    parser.add_argument('--threads', type=float, default=0.5, required=False, help='Directory to save model')
    parser.add_argument('--epochs', type=int,
                        default=50, required=False,
                        help='Directory to save model')
    parser.add_argument('--patience', type=int,
                        default=10, required=False,
                        help='Directory to save model')
    parser.add_argument('--num_labels', type=int,
                        default=4, required=False,
                        help='Directory to save model')
    parser.add_argument('--batch_size', type=int,
                        default=4, required=False,
                        help='Directory to save model')
    args = parser.parse_args()
    main(
        train_json=args.train_json,
        bert_model_path=args.bert_path,
        model_saved_path=args.model_saved_path,
        t=args.threads,
        epochs=args.epochs,
        patience=args.patience,
        val_json=args.val_json,
        test_json=args.test_json,
        num_labels=args.num_labels,
        batch_size=args.batch_size
    )
