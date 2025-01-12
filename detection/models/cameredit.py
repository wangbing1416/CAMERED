import os
import torch
import tqdm
import torch.nn as nn
from .layers import *
from transformers import BertModel


class SelfAttention(torch.nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = torch.nn.Linear(input_dim, input_dim)
        self.key = torch.nn.Linear(input_dim, input_dim)
        self.value = torch.nn.Linear(input_dim, input_dim)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attention_weights = self.softmax(attention_scores)
        output = torch.matmul(attention_weights, V)
        return torch.mean(output, dim=0)


class CameReditModel(torch.nn.Module):
    def __init__(self, pretrain_name, emb_dim, mlp_dims, dropout, category):
        super(CameReditModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_name).requires_grad_(False)

        for name, param in self.bert.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.fea_size = emb_dim
        self.att_neg = SelfAttention(emb_dim)
        self.att_pos = SelfAttention(emb_dim)
        self.fusion = nn.Sequential(
            nn.Linear(emb_dim * 4, emb_dim),
            nn.Tanh()
        )
        self.mlp = MLP(emb_dim * 3, [mlp_dims], output_dim=category, dropout=dropout)
        self.rnn = nn.GRU(input_size=emb_dim,
                          hidden_size=self.fea_size,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)
        self.attention = MaskAttention(emb_dim * 2)

    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        comment = kwargs['comment']
        comment_num = kwargs['comment_num']
        bert_feature = self.bert(inputs, attention_mask=masks)[0]
        comment_features = []
        for c in range(comment.shape[0]):  # average comment features
            max_len = comment_num[c] if comment_num[c] > 0 else 1
            temp_c_feature = self.bert(comment[c][:max_len])[0]
            comment_features.append(temp_c_feature[:, 0, :])

        news_feature = bert_feature[:, 0, :]
        cn_features = []
        for news, co in zip(news_feature, comment_features):
            cosine_similarities = F.cosine_similarity(co, news.unsqueeze(0), dim=-1)
            sorted_indices = torch.argsort(cosine_similarities, descending=True)
            split_point = len(co) // 2  # 平均分割点

            if co.shape[0] == 1: high_similarity_indices = sorted_indices[:split_point + 1]
            else: high_similarity_indices = sorted_indices[:split_point]
            co_high = co[high_similarity_indices]
            co_high = self.att_pos(co_high)

            low_similarity_indices = sorted_indices[split_point:]
            co_low = co[low_similarity_indices]
            co_low = self.att_neg(co_low)
            cn_features.append(self.fusion(torch.cat([co_high, co_high * co_low, co_high - co_low, co_low], dim=0)))

        # comment_feature = torch.stack(comment_features)
        cn_feature = torch.stack(cn_features)
        feature, _ = self.rnn(bert_feature)
        feature, _ = self.attention(feature, masks)
        output = self.mlp(torch.cat([feature, cn_feature], dim=1))
        return output.squeeze(1)
