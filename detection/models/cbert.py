import os
import torch
import tqdm
import torch.nn as nn
from .layers import *
from transformers import BertModel


class cBERTModel(torch.nn.Module):
    def __init__(self, pretrain_name, emb_dim, mlp_dims, dropout, category):
        super(cBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_name).requires_grad_(False)

        for name, param in self.bert.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.fea_size = emb_dim
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
        # emotion = kwargs['emotion']
        comment = kwargs['comment']
        comment_num = kwargs['comment_num']
        bert_feature = self.bert(inputs, attention_mask=masks)[0]
        comment_features = []
        for c in range(comment.shape[0]):  # average comment features
            max_len = comment_num[c] if comment_num[c] > 0 else 1
            temp_c_feature = self.bert(comment[c][:max_len])[0]
            comment_features.append(torch.mean(temp_c_feature[:, 0, :], dim=0))
        comment_feature = torch.stack(comment_features)
        feature, _ = self.rnn(bert_feature)
        feature, _ = self.attention(feature, masks)
        output = self.mlp(torch.cat([feature, comment_feature], dim=1))
        return output.squeeze(1)
