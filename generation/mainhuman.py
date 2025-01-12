import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from torch import nn
import argparse
import os
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers.modeling_outputs import BaseModelOutput
from data_loader import datamain2
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM

# 自定义模型，添加线性层
class T5WithLinearLayer(nn.Module):
    def __init__(self, model_name_or_path, device):
        super(T5WithLinearLayer, self).__init__()
        if "chinese" in model_name_or_path.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)

        for param in self.model.encoder.parameters():
            param.requires_grad = False
        for param in self.model.decoder.parameters():
            param.requires_grad = False

        hidden_size = self.model.config.d_model
        self.linear = nn.Linear(hidden_size, hidden_size).to(device)
        nn.init.eye_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)
        self.model.to(device)
        self.linear.to(device)
        self.device = device

    def get_decoder_hidden_states(self, input_ids, attention_mask):
        # 计算编码器输出
        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_hidden_states = encoder_outputs.last_hidden_state
        linear_output = self.linear(encoder_hidden_states)

        # 使用 input_ids 偏移作为 decoder_input_ids
        decoder_input_ids = self.model._shift_right(input_ids)
        decoder_outputs = self.model.decoder(input_ids=decoder_input_ids, encoder_hidden_states=linear_output,
                                             encoder_attention_mask=attention_mask)

        return decoder_outputs.last_hidden_state  # 返回隐藏状态

    def generateself(self, input_ids, attention_mask, **generation_args):
        max_length = generation_args.pop("max_length", 50)
        num_return_sequences = generation_args.pop("num_return_sequences", 3)
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)

        with torch.no_grad():
            encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            encoder_hidden_states = encoder_outputs.last_hidden_state
            linear_output = self.linear(encoder_hidden_states)
            encoder_outputs = BaseModelOutput(last_hidden_state=linear_output)

            generated_ids = self.model.generate(
                input_ids=input_ids,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                **generation_args
            )

        return generated_ids

    def forward(self, input_ids, attention_mask, target_ids):
        input_ids, attention_mask, target_ids = input_ids.to(self.device), attention_mask.to(self.device), target_ids.to(self.device)
        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_hidden_states = encoder_outputs.last_hidden_state
        linear_output = self.linear(encoder_hidden_states)

        decoder_input_ids = self.model._shift_right(target_ids)
        decoder_outputs = self.model.decoder(input_ids=decoder_input_ids, encoder_hidden_states=linear_output, encoder_attention_mask=attention_mask)

        sequence_output = decoder_outputs.last_hidden_state
        lm_logits = self.model.lm_head(sequence_output)
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), target_ids.view(-1))
        return loss

    @classmethod
    def load_linear(cls, model_name_or_path, linear_path, device):
        # 实例化对象
        instance = cls(model_name_or_path, device)

        # 加载线性层的权重
        linear_path = os.path.join(linear_path, "linear_layer.pt")
        instance.linear.load_state_dict(torch.load(linear_path, map_location=device))

        return instance


    def save_linear_layer(self, output_dir):
        # 创建保存路径
        os.makedirs(output_dir, exist_ok=True)
        # 保存线性层的权重
        linear_path = os.path.join(output_dir, 'linear_layer.pt')
        torch.save(self.linear.state_dict(), linear_path)
        print(f"Linear layer weights saved to {linear_path}")
    def save_checkpoint(self, optimizer, scheduler, epoch, output_dir, avg_loss):
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, 'checkpoint.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'avg_loss': avg_loss
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

def load_preprocessed_data(input_json_path, model_name_or_path):
    from data_loader import datamain
    dataset = datamain(input_json_path, model_name_or_path)
    return dataset
def evall(input_json_path, model_name_or_path, output_dir, resume=False, min_epochs=5, max_epochs=50, batch_size=4, learning_rate=5e-5, patience=3, min_delta=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_preprocessed_data(input_json_path, model_name_or_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = T5WithLinearLayer.load_linear(model_name_or_path,output_dir,device)
    model.eval()  # 设置模型为评估模式
    total_loss = 0

    with torch.no_grad():  # 禁用梯度计算
        for batch in dataloader:
            input_ids, attention_mask, target_ids = batch["input_ids"].to(device), batch["attention_mask"].to(device), \
            batch["target_ids"].to(device)
            loss = model(input_ids, attention_mask, target_ids)
            print(loss)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Average Loss: {avg_loss}")
    return avg_loss

def main(input_json_path, model_name_or_path, output_dir, resume=False, min_epochs=5, max_epochs=50, batch_size=4, learning_rate=5e-5, patience=3, min_delta=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_preprocessed_data(input_json_path, model_name_or_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = T5WithLinearLayer(model_name_or_path, device)
    optimizer = AdamW(model.linear.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, min_lr=1e-6, verbose=True)

    start_epoch = 0
    best_loss = float('inf')
    stop_training = False

    # 尝试加载中断点
    checkpoint_path = os.path.join(output_dir, 'checkpoint.pt')
    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['avg_loss']
        print(f"Resumed training from epoch {start_epoch}")
    elif resume:
        print("No checkpoint found, starting from scratch.")

    model.train()
    for epoch in range(start_epoch, max_epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids, attention_mask, target_ids = batch["input_ids"], batch["attention_mask"], batch["target_ids"]

            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, target_ids)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{max_epochs}, Average Loss: {avg_loss}")

        model.save_checkpoint(optimizer, scheduler, epoch, output_dir, avg_loss)
        scheduler.step(avg_loss)

        # Early stopping
        if epoch >= min_epochs:
            loss_reduction = best_loss - avg_loss
            if loss_reduction < min_delta:
                print(f"Early stopping at epoch {epoch + 1}. No significant loss improvement.")
                stop_training = True
            else:
                best_loss = avg_loss

        if stop_training:
            break

    model.save_linear_layer(output_dir)

def mainhuman(data, model_name_or_path, output_dir, resume=False, min_epochs=5, max_epochs=50, batch_size=4, learning_rate=5e-5):
    patience = 3
    min_delta = 0.001
    dataset = datamain2(data,model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #dataset = load_preprocessed_data(input_json_path, model_name_or_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = T5WithLinearLayer(model_name_or_path, device)
    optimizer = AdamW(model.linear.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, min_lr=1e-6, verbose=True)

    start_epoch = 0
    best_loss = float('inf')
    stop_training = False

    # 尝试加载中断点
    checkpoint_path = os.path.join(output_dir, 'checkpoint.pt')
    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['avg_loss']
        print(f"Resumed training from epoch {start_epoch}")
    elif resume:
        print("No checkpoint found, starting from scratch.")

    model.train()
    for epoch in range(start_epoch, max_epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids, attention_mask, target_ids = batch["input_ids"], batch["attention_mask"], batch["target_ids"]

            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, target_ids)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{max_epochs}, Average Loss: {avg_loss}")

        model.save_checkpoint(optimizer, scheduler, epoch, output_dir, avg_loss)
        scheduler.step(avg_loss)

        # Early stopping
        if epoch >= min_epochs:
            loss_reduction = best_loss - avg_loss
            if loss_reduction < min_delta:
                print(f"Early stopping at epoch {epoch + 1}. No significant loss improvement.")
                stop_training = True
            else:
                best_loss = avg_loss

        if stop_training:
            break

    model.save_linear_layer(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train T5 model with linear layer and checkpoints")
    parser.add_argument('--input_json_path', type=str, required=False, default="/home/zhaobingrui/MoE/data/twitter15/processed_train.json")
    parser.add_argument('--model_name_or_path', type=str, required=False, default="/home/zhaobingrui/MoE/flan-t5-base/")
    parser.add_argument('--output_dir', type=str, required=False, default="/home/zhaobingrui/MoE/saved_human_zidai/")
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint if available')
    parser.add_argument('--min_epochs', type=int, default=5, help='Minimum epochs before early stopping')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for training')
    parser.add_argument('--patience', type=int, default=3, help='Patience for learning rate scheduler')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum loss improvement to avoid early stopping')

    args = parser.parse_args()
    #main(args.input_json_path, args.model_name_or_path, args.output_dir, args.resume, args.min_epochs, args.max_epochs, args.batch_size, args.learning_rate, args.patience, args.min_delta)
    evall(args.input_json_path, args.model_name_or_path, args.output_dir, args.resume, args.min_epochs, args.max_epochs,
         args.batch_size, args.learning_rate, args.patience, args.min_delta)
