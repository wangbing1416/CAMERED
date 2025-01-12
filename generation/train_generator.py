import argparse
import json
import random
from typing import List, Tuple, Union
from mainMoE import T5WithLoRALayer
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
from mainMoE import mainmoe
from mainhuman import mainhuman
from gan import maingan


def process_jsons(json1_path: str, json2_path: str) -> List[Tuple[str, str]]:
    """
    Process two JSON files and create shuffled pairs of content/tweet and comments.

    Args:
        json1_path: Path to first JSON file (line-by-line JSON objects)
        json2_path: Path to second JSON file (JSON array)

    Returns:
        List of tuples containing (content/tweet, comment) pairs
    """
    pairs = []

    # Process JSON1 (line-by-line format)
    with open(json1_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # Get content or tweet field
                text = data.get('content') or data.get('tweet', '')
                # Get comments
                comments = data.get('comments', [])
                # Create pairs
                pairs.extend([(text, comment) for comment in comments])
            except json.JSONDecodeError:
                continue
    print(len(pairs))
    # Process JSON2 (array format)
    with open(json2_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            # Extend pairs list with existing pairs
            pairs.extend(data)
        except json.JSONDecodeError:
            print("Error reading JSON2 file")
    print(len(pairs))
    # Shuffle pairs
    random.shuffle(pairs)

    return pairs


def process_json(json1_path: str) -> List[Tuple[str, str]]:
    """
    Process two JSON files and create shuffled pairs of content/tweet and comments.

    Args:
        json1_path: Path to first JSON file (line-by-line JSON objects)
        json2_path: Path to second JSON file (JSON array)

    Returns:
        List of tuples containing (content/tweet, comment) pairs
    """
    pairs = []

    # Process JSON1 (line-by-line format)
    with open(json1_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # Get content or tweet field
                text = data.get('content') or data.get('tweet', '')
                # Get comments
                comments = data.get('comments', [])
                # Create pairs
                pairs.extend([(text, comment) for comment in comments])
            except json.JSONDecodeError:
                continue
    # Shuffle pairs
    random.shuffle(pairs)
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train generator")
    parser.add_argument('--comments_json', type=str, required=False, default="../data/weibo16/train.json", help='Path to the first input JSON file')
    parser.add_argument('--description_json', type=str, required=False, default="../data/weibo16/entity_description.json", help='Path to the second input JSON file')
    parser.add_argument('--t5_model', type=str, required=False, default="../chinese-t5-base/", help='Path to pretrained model or model name')
    parser.add_argument('--output_dir', type=str, default="../generator_results/moe_result/epoch_20,K_16,T_0.5/", required=False, help='Directory to save model')
    parser.add_argument('--human_batch_size', type=int, default=8,required=False, help='Batch size for training')
    parser.add_argument('--generator_batch_size', type=int, default=8, required=False, help='Batch size for training')
    parser.add_argument('--gan_batch_size', type=int, default=8, required=False, help='Batch size for training')
    parser.add_argument('--human_learning_rate', type=float, default=5e-5,required=False, help='Learning rate for training')
    parser.add_argument('--generator_learning_rate', type=float, default=5e-5, required=False,
                        help='Learning rate for training')
    parser.add_argument('--gan_generator_learning_rate', type=float, default=5e-5, required=False,
                        help='Learning rate for training')
    parser.add_argument('--gan_discriminator_learning_rate', type=float, default=5e-5, required=False,
                        help='Learning rate for training')
    parser.add_argument('--num_experts', type=int, default=10,required=False, help='Number of experts')
    parser.add_argument('--K', type=int, default=16,required=False, help='Number of clusters for LoRA layer')
    parser.add_argument('--threads', type=float, default=0.5, required=False, help='Number of threads for multi-processing')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--generator_max_epochs', type=int, default=1, required=False, help='Maximum number of training epochs')
    parser.add_argument('--generator_min_epochs', type=int, default=1, required=False,
                        help='Maximum number of training epochs')
    parser.add_argument('--human_max_epochs', type=int, default=1,required=False, help='Maximum number of training epochs')
    parser.add_argument('--human_min_epochs', type=int, default=1, required=False,
                        help='Maximum number of training epochs')
    parser.add_argument('--gan_epochs', type=int, default=5, required=False,
                        help='Maximum number of training epochs')
    parser.add_argument('--human_saved_path', type=str, required=False,
                        default="../data/weibo16/human/",
                        help='Path to the first input JSON file')
    parser.add_argument('--gan_saved_path', type=str, required=False,
                        default="../data/weibo16/gan/",
                        help='Path to the second input JSON file')

    args = parser.parse_args()
    data = process_jsons(args.comments_json, args.description_json)

    # train generator
    mainmoe(data, args.t5_model, args.output_dir, args.generator_batch_size, args.generator_learning_rate,
            args.num_experts, args.K, args.threads, False, args.generator_max_epochs, args.generator_min_epochs)
    data_human = process_json(args.comments_json)
    mainhuman(data, args.t5_model, args.human_saved_path, False, args.generator_min_epochs, args.generator_max_epochs,
              args.generator_batch_size, args.generator_learning_rate)
    maingan(args.comments_json, args.gan_epochs, args.gan_generator_learning_rate, args.gan_discriminator_learning_rate,
            args.t5_model, args.human_saved_path, args.output_dir, args.K, args.num_experts, args.threads,
            args.gan_saved_path, False)





