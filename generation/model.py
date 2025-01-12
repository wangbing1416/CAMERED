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


# 设置多进程启动方式为 'spawn'（必须在 __main__ 保护下）
def set_spawn_start_method():
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 如果已经设置，则忽略


# 确保设备可用
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
        random.shuffle(comments)  # 随机打乱评论
        return label, tweet, comments


def collate_fn(batch: List[Tuple[int, str, List[str]]], tokenizer: BertTokenizer, max_length: int):
    """
    自定义的collate函数，在数据加载阶段进行分词。
    不移动数据到 GPU，保持在 CPU 上。
    """
    labels, tweets, comments = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)

    # 编码tweets
    encoded_tweets = tokenizer(
        list(tweets),
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=max_length,
        return_token_type_ids=True  # 确保生成 token_type_ids
    )

    # 编码comments
    all_comments = [comment for comments_per_sample in comments for comment in comments_per_sample]
    if all_comments:
        encoded_comments = tokenizer(
            all_comments,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=max_length,
            return_token_type_ids=True  # 确保生成 token_type_ids
        )
    else:
        # 如果没有评论，创建空的张量
        encoded_comments = {
            'input_ids': torch.empty((0, max_length), dtype=torch.long),
            'attention_mask': torch.empty((0, max_length), dtype=torch.long),
            'token_type_ids': torch.empty((0, max_length), dtype=torch.long)
        }

    return labels, encoded_tweets, comments, encoded_comments


class CollateFnWrapper:
    """
    可序列化的collate_fn包装器类，避免使用lambda函数。
    """

    def __init__(self, tokenizer: BertTokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Tuple[int, str, List[str]]]):
        return collate_fn(batch, self.tokenizer, self.max_length)


class MyModel(nn.Module):
    def __init__(self, bert_model_path: str, t: float, num_labels: int = 4, hidden_size: int = 768,
                 max_length: int = 128):
        super(MyModel, self).__init__()
        self.t = t  # 相似度阈值
        self.num_labels = num_labels  # 分类数量
        self.hidden_size = hidden_size
        self.max_length = max_length

        # 加载标准的 BERT 模型
        self.bert = BertModel.from_pretrained(bert_model_path)

        # 自注意力网络
        self.attention_plus = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)
        self.attention_minus = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)

        # 可学习的投影矩阵 W
        self.W = nn.Linear(4 * hidden_size, hidden_size)

        # 分类器，动态适应不同的分类数量
        intermediate_size = hidden_size // 2  # 中间层的维度

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, intermediate_size),
            nn.BatchNorm1d(intermediate_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(intermediate_size, self.num_labels)  # 最后一层输出维度为 num_labels
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
    print(f"模型已保存至 {save_path}")


def load_model(model: nn.Module, optimizer: torch.optim.Optimizer, load_path: str) -> Tuple[
    nn.Module, torch.optim.Optimizer, int]:
    if os.path.exists(load_path):
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"模型已从 {load_path} 加载，继续从第 {start_epoch} 轮开始训练")
        return model, optimizer, start_epoch
    else:
        print(f"未找到模型文件 {load_path}")
        return model, optimizer, 0


def evaluate_model(model: nn.Module, dataloader: DataLoader) -> float:
    model.eval()  # 切换到评估模式
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            labels, encoded_tweets, comments, encoded_comments = batch
            # 将数据移动到 GPU
            labels = labels.to(device)
            for k in encoded_tweets:
                encoded_tweets[k] = encoded_tweets[k].to(device)
            for k in encoded_comments:
                if encoded_comments[k].numel() > 0:
                    encoded_comments[k] = encoded_comments[k].to(device)

            with autocast():
                outputs = model(encoded_tweets, encoded_comments, comments)  # [batch_size, num_labels]

            _, preds = torch.max(outputs, dim=1)  # 获取预测的类别索引

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"模型在验证集上的准确率为: {accuracy * 100:.2f}%")
    return accuracy


def setup_logger() -> logging.Logger:
    # 配置日志，输出到 best_accuracies.txt 和控制台
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler("best_accuracies.txt"),  # 输出到同一个文件
            logging.StreamHandler()  # 同时输出到控制台
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
    # 配置参数
    logger = setup_logger()
    logger.info(f"开始训练模型，t={t}")
    batch_size = 2  # 根据显存情况调整
    num_epochs = epochs
    best_accuracy = 0
    best_epoch = 0
    no_improvement_count = 0

    # 初始化 tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)

    # --- 训练和验证阶段 ---
    # 加载训练数据
    train_dataset = MyDataset(train_json, tokenizer, max_length=64)  # 根据需求调整 max_length
    collate_wrapper = CollateFnWrapper(tokenizer, max_length=64)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_wrapper,
        drop_last=True,
        num_workers=4,  # 根据系统资源调整
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 加载验证数据
    val_dataset = MyDataset(val_json, tokenizer, max_length=64)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_wrapper,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 初始化模型
    model = MyModel(bert_model_path, t, num_labels).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {'params': model.attention_plus.parameters(), 'lr': 1e-5},
        {'params': model.attention_minus.parameters(), 'lr': 1e-5},
        {'params': model.W.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-5}
    ])

    # 定义学习率调度器
    num_training_steps = num_epochs * len(train_dataloader)
    num_warmup_steps = int(0.02 * num_training_steps)  # 2% 的steps用于warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # 初始化 GradScaler
    scaler = GradScaler()  # 不传递 device_type，兼容旧版本 PyTorch

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            labels, encoded_tweets, comments, encoded_comments = batch

            # 将数据移动到 GPU
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

            # 释放未使用的缓存内存
            torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1}, 平均损失: {avg_loss:.4f}")

        # 评估模型
        accuracy = evaluate_model(model, val_dataloader)
        logger.info(f"Epoch {epoch + 1}, 验证集准确率: {accuracy * 100:.2f}%")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch + 1
            no_improvement_count = 0
            save_model(model, optimizer, epoch, model_saved_path)
            logger.info(f"保存最佳模型至 {model_saved_path} (Epoch {epoch + 1})")
        else:
            no_improvement_count += 1

        # 提前停止条件
        if no_improvement_count >= patience:
            logger.info(f"提前停止训练。最佳 Epoch: {best_epoch}, 最佳准确率: {best_accuracy * 100:.2f}%")
            break

    logger.info(f"训练完成。最佳 Epoch: {best_epoch}, 最佳准确率: {best_accuracy * 100:.2f}%\n")

    # --- 释放训练和验证数据 ---
    del train_dataset
    del train_dataloader
    del val_dataset
    del val_dataloader
    del model
    torch.cuda.empty_cache()

    # --- 测试阶段 ---
    # 重新加载模型
    model = MyModel(bert_model_path, t, num_labels=4).to(device)
    optimizer = torch.optim.Adam([
        {'params': model.attention_plus.parameters(), 'lr': 1e-5},
        {'params': model.attention_minus.parameters(), 'lr': 1e-5},
        {'params': model.W.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-5}
    ])
    # 加载最佳模型权重
    checkpoint = torch.load(model_saved_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()

    # 加载测试数据
    test_dataset = MyDataset(test_json, tokenizer, max_length=128)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_wrapper,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 评估模型在测试集上的性能
    test_accuracy = evaluate_model(model, test_dataloader)
    print(f"测试集准确率: {test_accuracy * 100:.2f}%")

    # 记录每个 `t` 的最佳准确率
    with open("best_accuracies.txt", "a") as f:
        f.write(f"Best Epoch: {best_epoch}, Best Accuracy: {best_accuracy * 100:.2f}%\n")

    # --- 释放测试数据 ---
    del test_dataset
    del test_dataloader
    del model
    torch.cuda.empty_cache()



if __name__ == "__main__":
    import argparse

    # 设置多进程启动方式为 'spawn'（必须在 __main__ 保护下）
    set_spawn_start_method()

    parser = argparse.ArgumentParser(description="Train generator"
                                     )
    parser.add_argument('--train_json', type=str, required=False,
                        default="/home/zhaobingrui/MoE/processed_data/twitter16/train.json",
                        help='Path to the first input JSON file')
    parser.add_argument('--val_json', type=str, required=False,
                        default="/home/zhaobingrui/MoE/processed_data/twitter16/val.json",
                        help='Path to the first input JSON file')
    parser.add_argument('--test_json', type=str, required=False,
                        default="/home/zhaobingrui/MoE/processed_data/twitter16/test.json",
                        help='Path to the first input JSON file')

    parser.add_argument('--bert_path', type=str, required=False, default="/home/zhaobingrui/MoE/bert/",
                        help='Path to the second input JSON file')
    parser.add_argument('--model_saved_path', type=str, required=False,
                        default="/home/zhaobingrui/MoE/processed_data/twitter16/final_result/",
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
