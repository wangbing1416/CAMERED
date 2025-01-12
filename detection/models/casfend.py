import os
import torch
import tqdm
import torch.nn as nn
from .layers import *
from transformers import BertModel


import torch
import torch.nn as nn
import torch.nn.functional as F


class CoAttentionLayer(nn.Module):
    def __init__(self):
        super(CoAttentionLayer, self).__init__()

    def forward(self, A, B):
        """
        input:
            A: Tensor of shape (N, L_A, D)
            B: Tensor of shape (N, L_B, D)
        return:
            A_co_attended: Tensor of shape (N, L_A, D)
            B_co_attended: Tensor of shape (N, L_B, D)
        """
        similarity_matrix = torch.matmul(A, B.transpose(2, 1))  # (N, L_A, L_B)

        attention_A_to_B = F.softmax(similarity_matrix, dim=-1)  # (N, L_A, L_B)
        attention_B_to_A = F.softmax(similarity_matrix.transpose(2, 1), dim=-1)  # (N, L_B, L_A)

        A_co_attended = torch.matmul(attention_A_to_B, B)  # (N, L_A, D)
        B_co_attended = torch.matmul(attention_B_to_A, A)  # (N, L_B, D)

        return torch.mean(A_co_attended, dim=1), torch.mean(B_co_attended, dim=1)


class CASFENDModel(torch.nn.Module):
    def __init__(self, pretrain_name, emb_dim, emo_dim, mlp_dims, dropout, category):
        super(CASFENDModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_name).requires_grad_(False)

        for name, param in self.bert.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.fea_size = emb_dim
        self.weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        self.co_attention = CoAttentionLayer()
        self.mlp = MLP(emb_dim * 1, [mlp_dims], output_dim=category, dropout=dropout)
        self.emotion_mlp = nn.Sequential(
            nn.Linear(emo_dim, emb_dim),
            nn.Tanh()
        )
        self.rnn = nn.GRU(input_size=emb_dim,
                          hidden_size=self.fea_size,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)
        self.attention = MaskAttention(emb_dim * 2)

    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        emotion = kwargs['emotion']
        comment = kwargs['comment']
        comment_num = kwargs['comment_num']
        bert_feature = self.bert(inputs, attention_mask=masks)[0]
        comment_features = []
        for c in range(comment.shape[0]):  # average comment features
            max_len = comment_num[c] if comment_num[c] > 0 else 1
            temp_c_feature = self.bert(comment[c][:max_len])[0]
            comment_features.append(torch.mean(temp_c_feature, dim=0))
        comment_feature = torch.stack(comment_features)

        emotion_feature = self.emotion_mlp(emotion)
        co_feature_A, co_feature_B = self.co_attention(bert_feature, comment_feature)
        feature, _ = self.rnn(bert_feature)
        feature, _ = self.attention(feature, masks)
        output = self.mlp(self.weights[0] * co_feature_A + self.weights[1] * co_feature_B + self.weights[2] * emotion_feature)
        return output.squeeze(1)


