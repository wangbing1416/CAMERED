import torch
import random
import tqdm
import re
import pandas as pd
import json
import numpy as np
import nltk
from transformers import BertTokenizer, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader

label_dict = {
    "real": 0,
    "fake": 1
}


def clean_tweet(tweet):
    # Replace URLs with special token
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", " ", tweet)
    # Replace mentions with 'user'
    text = re.sub(r"@\w+", " ", text)
    # Remove new lines
    text = text.replace("\n", " ")
    # Replace contractions and special characters
    text = text.replace("\'s", "")
    text = text.replace("\'", "")
    text = text.replace("n\'t", " n\'t")
    text = text.replace("#", "")
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace("&amp;", "")
    text = text.replace("&gt;", "")
    text = text.replace("\"", "")
    text = re.sub(r'\s+', ' ', text)
    # Remove leading and trailing whitespace
    text = text.strip()
    return text


def word2input(texts, entities, comments, g_comments,
               max_len, max_k, max_m, tokenizer, path):
    global c
    token_ids = []
    entity_ids = []
    comment_ids = []
    comment_num = []
    print("\nData Processing: tokenizing text from {}".format(path))
    # tokenizing text
    for text, entity, comment, g_comment in tqdm.tqdm(zip(texts, entities, comments, g_comments)):
        token_ids.append(tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length', truncation=True))
        entity_ids.append(tokenizer.encode(entity, max_length=8, add_special_tokens=True, padding='max_length', truncation=True))

        temp_comment_ids = [
            tokenizer.encode(
                comment[c] if c < len(comment) else '',
                max_length=int(max_len / 2),
                add_special_tokens=True, padding='max_length', truncation=True
            )
            for c in range(max_k)
        ]

        temp_comment_ids = temp_comment_ids[:min(len(comment), max_k)] + [
            tokenizer.encode(
                g_comment[g_c] if g_c < len(g_comment) else '',
                max_length=int(max_len / 2),
                add_special_tokens=True, padding='max_length', truncation=True
            )
            for g_c in range(max_m)
        ] + temp_comment_ids[min(len(comment), max_k):]

        comment_num.append(min(len(comment), max_k) + max_m)
        comment_ids.append(temp_comment_ids)

    token_ids = torch.tensor(token_ids)
    entity_ids = torch.tensor(entity_ids)
    comment_ids = torch.tensor(comment_ids)
    comment_num = torch.tensor(comment_num)
    masks = torch.zeros(token_ids.shape[0], token_ids.shape[1])
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i, :] = (tokens != mask_token_id)
    return token_ids, masks, entity_ids, comment_ids, comment_num


def get_twitter_dataloader(path, emo_path, entity_path, generated_path,
                           max_len, max_k, max_m,
                           batch_size, shuffle, pretrain_name):
    data_list = []
    with open(path, 'r', encoding='utf-8') as f_d:  # read data json
        for line in f_d:
            data_list.append(json.loads(line.strip()))

    with open(entity_path, 'r', encoding='utf-8') as f_e:  # read entity json
        entity = []
        entity_data = json.load(f_e)
        for e in entity_data:
            entity.append(' '.join(e))

    with open(generated_path, 'r', encoding='utf-8') as f_c:  # read generated comments json
        generated_comments = json.load(f_c)

    df_data = pd.DataFrame(columns=('content', 'label', 'comments', 'g_comments'))
    print("\nData Processing: loading data from {}".format(path))
    comment_num = 0
    for item, g_comments in tqdm.tqdm(zip(data_list, generated_comments)):
        tmp_data = {}
        tmp_data['content'] = clean_tweet(item['tweets'][0])
        tmp_data['comments'] = [clean_tweet(s) for s in item['tweets'][1:]]
        tmp_data['g_comments'] = [clean_tweet(g_s) for g_s in g_comments]
        comment_num += len(tmp_data['comments'])

        if 'chinese' in pretrain_name: tmp_data['label'] = label_dict[item['label']]
        else: tmp_data['label'] = item['label']  # real-0, fake-1
        df_data = pd.concat([df_data, pd.DataFrame([tmp_data])], ignore_index=True)

    print("{} samples, average {:.4f} comments".format(len(data_list), comment_num / len(data_list)))
    emotion = np.load(emo_path).astype('float32')
    emotion = torch.tensor(emotion)
    content = df_data['content'].to_numpy()
    entity = np.array(entity)
    label = torch.tensor(df_data['label'].astype(int).to_numpy())

    tokenizer = AutoTokenizer.from_pretrained(pretrain_name)
    # tokenize content, comments, generated comments into indices
    content_token_ids, content_masks, entity_ids, comment_token_ids, comment_num = word2input(content, entity, df_data['comments'], df_data['g_comments'],
                                                                                              max_len, max_k, max_m, tokenizer, path)

    dataset = TensorDataset(content_token_ids, content_masks, entity_ids, comment_token_ids, comment_num, emotion, label)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
        shuffle=shuffle
    )
    return dataloader


def get_weibo_dataloader(path, emo_path, entity_path, generated_path,
                         max_len, max_k, max_m,
                         batch_size, shuffle, pretrain_name):
    with open(path, 'r', encoding='utf-8') as f_d:  # read data json
        data_list = json.load(f_d)

    with open(entity_path, 'r', encoding='utf-8') as f_e:  # read entity json
        entity = []
        entity_data = json.load(f_e)
        for e in entity_data:
            entity.append(' '.join(e))

    with open(generated_path, 'r', encoding='utf-8') as f_c:  # read generated comments json
        generated_comments = json.load(f_c)

    df_data = pd.DataFrame(columns=('content', 'label', 'comments', 'g_comments'))
    print("\nData Processing: loading data from {}".format(path))
    comment_num = 0
    for item, g_comments in tqdm.tqdm(zip(data_list, generated_comments)):
        tmp_data = {}
        tmp_data['content'] = clean_tweet(item['content'])
        tmp_data['comments'] = [clean_tweet(s) for s in item['comments']]
        tmp_data['g_comments'] = [clean_tweet(g_s) for g_s in g_comments]
        comment_num += len(tmp_data['comments'])

        if 'chinese' in pretrain_name: tmp_data['label'] = label_dict[item['label']]
        else: tmp_data['label'] = item['label']  # real-0, fake-1
        df_data = pd.concat([df_data, pd.DataFrame([tmp_data])], ignore_index=True)

    print("{} samples, average {:.4f} comments".format(len(data_list), comment_num / len(data_list)))
    emotion = np.load(emo_path).astype('float32')
    emotion = torch.tensor(emotion)
    content = df_data['content'].to_numpy()
    entity = np.array(entity)
    label = torch.tensor(df_data['label'].astype(int).to_numpy())

    tokenizer = AutoTokenizer.from_pretrained(pretrain_name)
    # tokenize content, comments, generated comments into indices
    content_token_ids, content_masks, entity_ids, comment_token_ids, comment_num = word2input(content, entity, df_data['comments'], df_data['g_comments'],
                                                                                              max_len, max_k, max_m, tokenizer, path)

    dataset = TensorDataset(content_token_ids, content_masks, entity_ids, comment_token_ids, comment_num, emotion, label)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
        shuffle=shuffle
    )
    return dataloader
