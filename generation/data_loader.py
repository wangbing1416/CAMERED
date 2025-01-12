import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer


# define Dataset class
class NewsCommentDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        original_tweet, comment = self.pairs[idx]
        inputs = self.tokenizer.encode_plus(
            original_tweet,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        targets = self.tokenizer.encode_plus(
            comment,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_ids": target_ids
        }


# main
def datamain(preprocessed_json_path, model_name='t5-base'):
    # load preprocessed data
    with open(preprocessed_json_path, 'r', encoding='utf-8') as f:
        pairs = json.load(f)

    # load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # create Dataset class
    dataset = NewsCommentDataset(pairs, tokenizer)
    return dataset


def datamain2(data, model_name):
    # load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # create Dataset class
    dataset = NewsCommentDataset(data, tokenizer)
    return dataset
