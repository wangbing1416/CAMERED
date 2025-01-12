import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
from mainhuman import T5WithLinearLayer
from mainMoE import T5WithLoRALayer
import os
# 自定义数据集类
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            #"target_ids": encoding["input_ids"].squeeze()  # 使用自回归目标
        }

# 判别器类
class AdvancedDiscriminator(nn.Module):
    def __init__(self, hidden_size):
        super(AdvancedDiscriminator, self).__init__()

        # 使用多层线性层
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )

        # 自注意力层
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4)

    def forward(self, x):
        x = x.transpose(0, 1)  # [seq_len, batch_size, hidden_size]
        x, _ = self.attention(x, x, x)
        x = x.transpose(0, 1)  # 恢复到 [batch_size, seq_len, hidden_size]
        x = x.mean(dim=1)  # 平均池化到 [batch_size, hidden_size]
        return self.fc_layers(x)

# 对抗训练循环
def adversarial_training_loop(generator, human_model, discriminator, dataloader, device, epochs=10, generator_lr=1e-4, discriminator_lr=1e-4, output_dir='./checkpoints', start_epoch=0):
    generator_optimizer = AdamW(generator.parameters(), lr=generator_lr)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=discriminator_lr)
    adversarial_loss = nn.BCELoss()

    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在

    for epoch in range(start_epoch, epochs):
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # 训练生成器
            generator.train()
            discriminator.eval()

            fake_hidden_states = generator.get_decoder_hidden_states(input_ids, attention_mask)
            real_labels = torch.ones(fake_hidden_states.size(0), 1).to(device)
            generator_adv_loss = adversarial_loss(discriminator(fake_hidden_states), real_labels)
            total_generator_loss = generator_adv_loss

            generator_optimizer.zero_grad()
            total_generator_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            generator_optimizer.step()

            # 每20个batch更新一次判别器
            if i % 20 == 0:
                generator.eval()
                human_model.eval()
                discriminator.train()

                fake_hidden_states = generator.get_decoder_hidden_states(input_ids, attention_mask)
                fake_labels = torch.zeros(fake_hidden_states.size(0), 1).to(device)

                with torch.no_grad():
                    real_hidden_states = human_model.get_decoder_hidden_states(input_ids, attention_mask)
                real_labels = torch.ones(real_hidden_states.size(0), 1).to(device)

                real_loss = adversarial_loss(discriminator(real_hidden_states), real_labels)
                fake_loss = adversarial_loss(discriminator(fake_hidden_states.detach()), fake_labels)
                discriminator_loss = (real_loss + fake_loss) / 2

                discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                discriminator_optimizer.step()

            print(f"Epoch {epoch + 1}, Batch {i + 1}, D Loss: {discriminator_loss.item()}, G Loss: {total_generator_loss.item()}")

        # 每5个 epoch 保存一次 checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'generator_optimizer_state_dict': generator_optimizer.state_dict(),
                'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict()
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
# 主函数
def main(output_dir='./checkpoints', resume=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    json_path = "../data/twitter15/train_processed_tweetonly.json"
    with open(json_path, "r") as f:
        texts = json.load(f)
    epochs = 5
    generator_lr = 1e-4
    discriminator_lr = 1e-4

    # 初始化模型和判别器
    human_model = T5WithLinearLayer.load_linear(
        "../flan-t5-base/",
        "../saved_human_jc/",
        device
    )
    generator = T5WithLoRALayer.load_linear_layer(
        "../flan-t5-base/",
        "../saved_mix_linear_jc/",
        device,
        num_experts=10,
        K=16,
        threshold=0.5
    ).to(device)
    discriminator = AdvancedDiscriminator(hidden_size=generator.model.config.d_model).to(device)

    # 加载数据
    tokenizer = human_model.tokenizer
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    start_epoch = 0
    if resume:
        checkpoints = [f for f in os.listdir(output_dir) if f.startswith('checkpoint')]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda f: int(f.split('_')[-1].split('.')[0]))
            checkpoint = torch.load(os.path.join(output_dir, latest_checkpoint))
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Resuming training from checkpoint: {latest_checkpoint}, starting at epoch {start_epoch}")
        else:
            print("No checkpoint found. Starting training from scratch.")

    # 开始对抗训练
    adversarial_training_loop(
        generator, human_model, discriminator, dataloader, device, epochs, generator_lr, discriminator_lr, output_dir, start_epoch=start_epoch
    )

    # 最后保存模型
    generator_save_path = os.path.join(output_dir, "generator")
    discriminator_save_path = os.path.join(output_dir, "final_discriminator.pth")
    generator.save_linear_layer(generator_save_path)
    torch.save(discriminator.state_dict(), discriminator_save_path)
    print(f"Generator saved to {generator_save_path}")
    print(f"Discriminator saved to {discriminator_save_path}")

def maingan(jsons,epochs,generator_lr,discriminator_lr,t5_model,human_saved_path,generator_saved_path,K,experts,threshold,output_dir='./checkpoints', resume=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    texts = extract_content(jsons)



    # 初始化模型和判别器
    human_model = T5WithLinearLayer.load_linear(
        t5_model,
        human_saved_path,
        device
    )
    generator = T5WithLoRALayer.load_linear_layer(
        t5_model,
        generator_saved_path,
        device,
        experts,
        K,
        threshold
    ).to(device)
    discriminator = AdvancedDiscriminator(hidden_size=generator.model.config.d_model).to(device)

    # 加载数据
    tokenizer = human_model.tokenizer
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    start_epoch = 0
    if resume:
        checkpoints = [f for f in os.listdir(output_dir) if f.startswith('checkpoint')]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda f: int(f.split('_')[-1].split('.')[0]))
            checkpoint = torch.load(os.path.join(output_dir, latest_checkpoint))
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Resuming training from checkpoint: {latest_checkpoint}, starting at epoch {start_epoch}")
        else:
            print("No checkpoint found. Starting training from scratch.")

    # 开始对抗训练
    adversarial_training_loop(
        generator, human_model, discriminator, dataloader, device, epochs, generator_lr, discriminator_lr, output_dir, start_epoch=start_epoch
    )

    # 最后保存模型
    generator_save_path = os.path.join(output_dir, "final_generator")
    discriminator_save_path = os.path.join(output_dir, "final_discriminator.pth")
    os.makedirs(generator_save_path, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    generator.save_linear_layer(generator_save_path)
    torch.save(discriminator.state_dict(), discriminator_save_path)
    print(f"Generator saved to {generator_save_path}")
    print(f"Discriminator saved to {discriminator_save_path}")

def extract_content(filename):
    content_list = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            content = data.get('tweet') or data.get('content')
            if content:
                content_list.append(content)
    return content_list

def save_checkpoint_as_final_models(output_dir, checkpoint_path):
    # 加载 checkpoint 文件
    checkpoint = torch.load(checkpoint_path)

    # 提取生成器和判别器的状态字典
    generator_state_dict = checkpoint['generator_state_dict']
    discriminator_state_dict = checkpoint['discriminator_state_dict']

    # 定义保存路径
    generator_save_path = os.path.join(output_dir, "final_generator")
    discriminator_save_path = os.path.join(output_dir, "final_discriminator.pth")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 初始化生成器
    generator = T5WithLoRALayer.load_linear_layers(
        model_name_or_path="../flan-t5-base/",
        linear_layer_path=None,  # 初始化时不加载已有的线性层权重
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_experts=10, K=16, threshold=0.5
    )
    generator.load_state_dict(generator_state_dict)  # 加载生成器权重
    generator.save_linear_layer(generator_save_path)  # 保存生成器的线性层

    # 保存判别器权重
    torch.save(discriminator_state_dict, discriminator_save_path)

    print(f"Generator saved to {generator_save_path}")
    print(f"Discriminator saved to {discriminator_save_path}")



if __name__ == "__main__":
    '''resume = False  # 修改为 True 或 False 来决定是否恢复训练
    output_dir = "../saved_gan_freeze/"
    main(output_dir=output_dir, resume=resume)'''
    output_dir = "../saved_gan_freeze/"
    checkpoint_path = "../saved_gan_freeze/checkpoint_epoch_5.pth"
    save_checkpoint_as_final_models(output_dir, checkpoint_path)
