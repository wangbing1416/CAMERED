import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from torch import nn
import argparse
import os
from transformers.modeling_outputs import BaseModelOutput
import itertools
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM

from data_loader import datamain
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from torch import nn
import argparse
import os
from transformers.modeling_outputs import BaseModelOutput
import itertools
import gc
from torch.optim.lr_scheduler import ReduceLROnPlateau
torch.autograd.set_detect_anomaly(True)
from data_loader import datamain2


class T5WithLoRALayer(nn.Module):
    def __init__(self, model_name_or_path, device, num_experts, K, threads):
        super(T5WithLoRALayer, self).__init__()
        if "chinese" in model_name_or_path.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.threads = threads

        for param in self.model.encoder.parameters():
            param.requires_grad = False
        for param in self.model.decoder.parameters():
            param.requires_grad = False

        hidden_size = self.model.config.d_model
        self.router = nn.Linear(hidden_size, num_experts).to(device)

        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_size, K).to(device),
            nn.Linear(K, hidden_size).to(device)
        ) for _ in range(num_experts)])

        for expert in self.experts:
            nn.init.kaiming_uniform_(expert[0].weight, nonlinearity='relu')
            nn.init.kaiming_uniform_(expert[1].weight, nonlinearity='relu')
            nn.init.constant_(expert[0].bias, 0)
            nn.init.constant_(expert[1].bias, 0)

        nn.init.kaiming_uniform_(self.router.weight, nonlinearity='relu')
        nn.init.constant_(self.router.bias, 0)

        self.model.to(device)
        self.device = device

    def normalize_similarity_matrix(self, similarity_matrix):
        min_val = similarity_matrix.min()
        max_val = similarity_matrix.max()
        normalized_matrix = (similarity_matrix - min_val) / (max_val - min_val)
        return normalized_matrix

    def forward(self, input_ids, attention_mask, target_ids):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        target_ids = target_ids.to(self.device)

        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_hidden_states = encoder_outputs.last_hidden_state

        routing_weights = self.router(encoder_hidden_states)  # [batch_size, seq_len, num_experts]
        routing_weights = torch.softmax(routing_weights, dim=-1)
        expert_outputs = [expert(encoder_hidden_states) for expert in self.experts]

        similarity_matrix = self.compute_similarity_matrix(expert_outputs)
        normalized_similarity_matrix = self.normalize_similarity_matrix(similarity_matrix)

        # grouping
        expert_groups = self.partition_experts(normalized_similarity_matrix, self.threads)

        max_weight_sum = -float('inf')
        best_group = None
        for group in expert_groups:
            weight_sum = sum(routing_weights[..., i].sum() for i in group)
            if weight_sum > max_weight_sum:
                max_weight_sum = weight_sum
                best_group = group
        for i, expert in enumerate(self.experts):
            requires_grad = i in best_group
            for param in expert.parameters():
                param.requires_grad = requires_grad

        # weighting experts
        selected_weights = torch.stack([routing_weights[..., i] for i in best_group], dim=-1)
        normalized_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True)

        # normalizing
        combined_output_weighted = sum(normalized_weights[..., idx].unsqueeze(-1) * expert_outputs[i]
                                       for idx, i in enumerate(best_group))

        decoder_input_ids = self.model._shift_right(target_ids)
        decoder_outputs = self.model.decoder(input_ids=decoder_input_ids, encoder_hidden_states=combined_output_weighted,
                                             encoder_attention_mask=attention_mask)

        sequence_output = decoder_outputs.last_hidden_state
        lm_logits = self.model.lm_head(sequence_output)
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), target_ids.view(-1))

        return loss

    def get_decoder_hidden_states(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # encoding
        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_hidden_states = encoder_outputs.last_hidden_state

        # weighting
        routing_weights = self.router(encoder_hidden_states)  # [batch_size, seq_len, num_experts]
        routing_weights = torch.softmax(routing_weights, dim=-1)

        expert_outputs = [expert(encoder_hidden_states) for expert in self.experts]

        # similarity and normalizing
        similarity_matrix = self.compute_similarity_matrix(expert_outputs)
        normalized_similarity_matrix = self.normalize_similarity_matrix(similarity_matrix)

        # grouping
        expert_groups = self.partition_experts(normalized_similarity_matrix, self.threads)

        max_weight_sum = -float('inf')
        best_group = None
        for group in expert_groups:
            weight_sum = sum(routing_weights[..., i].sum() for i in group)
            if weight_sum > max_weight_sum:
                max_weight_sum = weight_sum
                best_group = group

        # requires_grad=True
        for i, expert in enumerate(self.experts):
            requires_grad = i in best_group
            for param in expert.parameters():
                param.requires_grad = requires_grad

        # weighting
        # combined_output_weighted = sum(routing_weights[..., i].unsqueeze(-1) * expert_outputs[i] for i in best_group)
        selected_weights = torch.stack([routing_weights[..., i] for i in best_group], dim=-1)
        normalized_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True)

        # normalizing combined_output_weighted
        combined_output_weighted = sum(normalized_weights[..., idx].unsqueeze(-1) * expert_outputs[i]
                                       for idx, i in enumerate(best_group))

        # decoding
        decoder_input_ids = self.model._shift_right(input_ids)
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=combined_output_weighted,
            encoder_attention_mask=attention_mask,
            return_dict=True
        )

        return decoder_outputs.last_hidden_state

    def compute_similarity_matrix(self, expert_outputs):
        num_experts = len(expert_outputs)
        similarity_matrix = torch.zeros((num_experts, num_experts)).to(self.device)
        for i, j in itertools.combinations(range(num_experts), 2):
            similarity = F.cosine_similarity(expert_outputs[i].view(-1), expert_outputs[j].view(-1), dim=0)
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
        return similarity_matrix

    def partition_experts(self, similarity_matrix, threshold):
        num_experts = similarity_matrix.size(0)
        groups = []
        visited = [False] * num_experts

        def dfs(node, group):
            visited[node] = True
            group.append(node)
            for neighbor in range(num_experts):
                if abs(similarity_matrix[node, neighbor]) > threshold and not visited[neighbor]:
                    dfs(neighbor, group)

        for expert in range(num_experts):
            if not visited[expert]:
                group = []
                dfs(expert, group)
                groups.append(group)
        return groups

    def generateselfs(self, input_ids, attention_mask, threads, **generation_args):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # encoding
        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_hidden_states = encoder_outputs.last_hidden_state

        # weighting
        routing_weights = self.router(encoder_hidden_states)
        routing_weights = torch.softmax(routing_weights, dim=-1)

        # output from experts
        expert_outputs = [expert(encoder_hidden_states) for expert in self.experts]

        # similarities
        similarity_matrix = self.compute_similarity_matrix(expert_outputs)
        normalized_similarity_matrix = self.normalize_similarity_matrix(similarity_matrix)

        # grouping
        expert_groups = self.partition_experts(normalized_similarity_matrix, threads)

        all_generated_ids = []

        for group in expert_groups:
            combined_output_weighted = sum(routing_weights[..., i].unsqueeze(-1) * expert_outputs[i]
                                           for i in group) / len(group)

            new_encoder_outputs = BaseModelOutput(last_hidden_state=combined_output_weighted)

            generated_ids = self.model.generate(
                input_ids=input_ids,
                encoder_outputs=new_encoder_outputs,
                attention_mask=attention_mask,
                **generation_args
            )

            all_generated_ids.append(generated_ids)

        return all_generated_ids

    def generateself(self, input_ids, attention_mask, threads, **generation_args):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_hidden_states = encoder_outputs.last_hidden_state

        routing_weights = self.router(encoder_hidden_states)
        routing_weights = torch.softmax(routing_weights, dim=-1)

        expert_outputs = [expert(encoder_hidden_states) for expert in self.experts]

        similarity_matrix = self.compute_similarity_matrix(expert_outputs)
        normalized_similarity_matrix = self.normalize_similarity_matrix(similarity_matrix)

        expert_groups = self.partition_experts(normalized_similarity_matrix, threads)

        max_weight_sum = -float('inf')
        best_group = None
        for group in expert_groups:
            weight_sum = sum(routing_weights[..., i].sum() for i in group)
            if weight_sum > max_weight_sum:
                max_weight_sum = weight_sum
                best_group = group

        # combined_output_weighted = sum(routing_weights[..., i].unsqueeze(-1) * expert_outputs[i] for i in best_group) / len(best_group)
        selected_weights = torch.stack([routing_weights[..., i] for i in best_group], dim=-1)
        normalized_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True)

        # weighting combined_output_weighted
        combined_output_weighted = sum(normalized_weights[..., idx].unsqueeze(-1) * expert_outputs[i]
                                       for idx, i in enumerate(best_group))

        new_encoder_outputs = BaseModelOutput(last_hidden_state=combined_output_weighted)

        generated_ids = self.model.generate(
            input_ids=input_ids,
            encoder_outputs=new_encoder_outputs,
            attention_mask=attention_mask,
            **generation_args
        )

        return generated_ids

    def save_linear_layer(self, save_directory):
        router_path = os.path.join(save_directory, 'router.pt')
        torch.save(self.router.state_dict(), router_path)
        print(f"Router weights saved to {router_path}")

        for i, linear in enumerate(self.experts):
            linear_1_path = os.path.join(save_directory, f'linear_layer_1_{i}.pt')
            linear_2_path = os.path.join(save_directory, f'linear_layer_2_{i}.pt')
            torch.save(linear[0].state_dict(), linear_1_path)
            torch.save(linear[1].state_dict(), linear_2_path)
            print(f"Linear layer {i} part 1 weights saved to {linear_1_path}")
            print(f"Linear layer {i} part 2 weights saved to {linear_2_path}")

    @classmethod
    def load_linear_layer(cls, model_name_or_path, linear_layer_path, device, num_experts, K, threshold):
        instance = cls(model_name_or_path, device, num_experts, K, threshold)
        instance.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        instance.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)

        # load routing
        router_path = os.path.join(linear_layer_path, 'router.pt')
        instance.router.load_state_dict(torch.load(router_path, map_location=device))
        instance.router.to(device)

        # load experts
        for i, expert in enumerate(instance.experts):
            linear_1_path = os.path.join(linear_layer_path, f'linear_layer_1_{i}.pt')
            linear_2_path = os.path.join(linear_layer_path, f'linear_layer_2_{i}.pt')
            expert[0].load_state_dict(torch.load(linear_1_path, map_location=device))
            expert[1].load_state_dict(torch.load(linear_2_path, map_location=device))
            expert.to(device)

        return instance


    @classmethod
    def load_linear_layers(cls, model_name_or_path, linear_layer_path, device, num_experts, K, threshold):
        instance = cls(model_name_or_path, device, num_experts, K, threshold)
        instance.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        instance.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)

        if linear_layer_path:
            router_path = os.path.join(linear_layer_path, 'router.pt')
            instance.router.load_state_dict(torch.load(router_path, map_location=device))
            instance.router.to(device)

            for i, expert in enumerate(instance.experts):
                linear_1_path = os.path.join(linear_layer_path, f'linear_layer_1_{i}.pt')
                linear_2_path = os.path.join(linear_layer_path, f'linear_layer_2_{i}.pt')
                expert[0].load_state_dict(torch.load(linear_1_path, map_location=device))
                expert[1].load_state_dict(torch.load(linear_2_path, map_location=device))
                expert.to(device)

        return instance


def load_preprocessed_data(input_json_path_1, input_json_path_2, model_name_or_path):
    dataset1 = datamain(input_json_path_1, model_name_or_path)
    dataset2 = datamain(input_json_path_2, model_name_or_path)

    combined_dataset = list(dataset1) + list(dataset2)
    random.shuffle(combined_dataset)

    return combined_dataset


def save_checkpoint(epoch, model, optimizer, output_dir):
    checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")


def load_checkpoint(model, optimizer, output_dir):
    checkpoints = [f for f in os.listdir(output_dir) if f.startswith('checkpoint')]
    if not checkpoints:
        print("No checkpoint found.")
        return 0

    latest_checkpoint = max(checkpoints, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    checkpoint = torch.load(os.path.join(output_dir, latest_checkpoint))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming from checkpoint {latest_checkpoint}, starting at epoch {start_epoch}")
    return start_epoch


def main(input_json_path_1, input_json_path_2, model_name_or_path, output_dir, batch_size, learning_rate,
         num_experts, K, threads, resume=False, min_epochs=5, max_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_preprocessed_data(input_json_path_1, input_json_path_2, model_name_or_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = T5WithLoRALayer(model_name_or_path, device, num_experts, K, threads)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Initialize scheduler for adaptive learning rate
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6, verbose=True)

    # Early stopping parameters
    early_stop_patience = 5
    no_improve_epochs = 0
    best_loss = float('inf')

    if resume and os.path.exists(output_dir):
        start_epoch = load_checkpoint(model, optimizer, output_dir)
    else:
        start_epoch = 0

    model.train()
    for epoch in range(start_epoch, max_epochs):
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["target_ids"].to(device)

            loss = model(input_ids, attention_mask, target_ids)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(dataloader)}, Batch Loss: {loss.item()}")

        # Calculate average loss and adjust learning rate
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)

        # Check for early stopping
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # Early stopping only kicks in after min_epochs
        if no_improve_epochs >= early_stop_patience and epoch + 1 >= min_epochs:
            print(f"Early stopping at epoch {epoch + 1} due to lack of improvement.")
            break

        # Save checkpoint after each epoch
        save_checkpoint(epoch, model, optimizer, output_dir)

        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")

        # Clean up memory
        torch.cuda.empty_cache()
        gc.collect()

    # Final model and tokenizer save
    model.save_linear_layer(output_dir)
    model.tokenizer.save_pretrained(output_dir)

    
def mainmoe(data, model_name_or_path, output_dir, batch_size, learning_rate,
         num_experts, K, threads, resume=False, min_epochs=5, max_epochs=50):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = datamain2(data,model_name_or_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = T5WithLoRALayer(model_name_or_path, device, num_experts, K, threads)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Initialize scheduler for adaptive learning rate
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6, verbose=True)

    # Early stopping parameters
    early_stop_patience = 5
    no_improve_epochs = 0
    best_loss = float('inf')

    if resume and os.path.exists(output_dir):
        start_epoch = load_checkpoint(model, optimizer, output_dir)
    else:
        start_epoch = 0

    model.train()
    for epoch in range(start_epoch, max_epochs):
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["target_ids"].to(device)

            loss = model(input_ids, attention_mask, target_ids)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(dataloader)}, Batch Loss: {loss.item()}")

        # Calculate average loss and adjust learning rate
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)

        # Check for early stopping
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # Early stopping only kicks in after min_epochs
        if no_improve_epochs >= early_stop_patience and epoch + 1 >= min_epochs:
            print(f"Early stopping at epoch {epoch + 1} due to lack of improvement.")
            break

        # Save checkpoint after each epoch
        save_checkpoint(epoch, model, optimizer, output_dir)

        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")

        # Clean up memory
        torch.cuda.empty_cache()
        gc.collect()

    # Final model and tokenizer save
    model.save_linear_layer(output_dir)
    model.tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train T5 model with linear layer and checkpointing")
    parser.add_argument('--input_json_path_1', type=str, required=False, default="../data/twitter15/processed_train.json", help='Path to the first input JSON file')
    parser.add_argument('--input_json_path_2', type=str, required=False, default="../data/twitter15/processed_tweet_descriptions.json", help='Path to the second input JSON file')
    parser.add_argument('--model_name_or_path', type=str, required=False, default="../flan-t5-base/", help='Path to pretrained model or model name')
    parser.add_argument('--output_dir', type=str, default="../saved_mix_linear_jc/", required=False, help='Directory to save model')
    parser.add_argument('--batch_size', type=int, default=4, required=False, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-5, required=False, help='Learning rate for training')
    parser.add_argument('--num_experts', type=int, default=10, required=False, help='Number of experts')
    parser.add_argument('--K', type=int, default=16, required=False, help='Number of clusters for LoRA layer')
    parser.add_argument('--threads', type=float, default=0.5, required=False, help='Number of threads for multi-processing')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--min_epochs', type=int, default=5, required=False, help='Minimum number of epochs before early stopping')
    parser.add_argument('--max_epochs', type=int, default=20, required=False, help='Maximum number of training epochs')
    args = parser.parse_args()

    main(args.input_json_path_1, args.input_json_path_2, args.model_name_or_path, args.output_dir,
         args.batch_size, args.learning_rate, args.num_experts, args.K, args.threads, args.resume,
         args.min_epochs, args.max_epochs)
