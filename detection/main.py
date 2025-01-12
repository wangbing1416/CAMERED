
import argparse

import torch
import os
import sys
import logging
import json
import datetime
import random
import numpy as np
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from trainer import Trainer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))  # logger output as print()

data_path_list = {
    'twitter15': '../data/Twitter15/',
    'twitter16': '../data/Twitter16/',
    'weibo16': '../data/Weibo16/',
    'weibo20': '../data/Weibo20/',
}

pretrained_model_list = {
    'twitter15': '../../huggingface/bert-base-uncased',
    'twitter16': '../../huggingface/bert-base-uncased',
    'weibo16': '../../huggingface/chinese-bert-wwm-ext',
    'weibo20': '../../huggingface/chinese-bert-wwm-ext',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='twitter16', help='twitter15, twitter16, weibo16, weibo20')
    parser.add_argument('--model', type=str, default='cameredit', help='cbert, bertemo, defend, casfend, kahan, cameredit')

    # parser.add_argument('--pretrain_name', type=str, default='../../huggingface/bert-base-uncased')
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--train_k', type=int, default=16)
    parser.add_argument('--test_k', type=int, default=2)
    parser.add_argument('--train_m', type=int, default=0)
    parser.add_argument('--test_m', type=int, default=0)
    parser.add_argument('--aug_prob', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--emb_dim', type=int, default=768)
    parser.add_argument('--inner_dim', type=int, default=384)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=7e-5)
    parser.add_argument('--mlp_lr', type=float, default=7e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for AdamW optimizer.")
    parser.add_argument('--save_param_dir', default='./param_model/')
    # parser.add_argument('--log_dir', default='./logs/')
    args = parser.parse_args()

    args.data_path = data_path_list[args.dataset]
    args.pretrain_name = pretrained_model_list[args.dataset]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.log_dir = './logs/' + args.dataset + '/'
    param_log_dir = args.log_dir
    if not os.path.exists(param_log_dir):
        os.makedirs(param_log_dir)
    nowtime = datetime.datetime.now().strftime("%m%d-%H%M")
    param_log_file = os.path.join(param_log_dir, args.model + '_' + args.dataset + '_seed' + str(args.seed) + '_' + nowtime + '.txt')
    logger.addHandler(logging.FileHandler(param_log_file))
    logger.info('> training arguments:')
    for arg in args._get_kwargs():
        logger.info('>>> {0}: {1}'.format(arg[0], arg[1]))

    trainer = Trainer(args)

    metrics, model_path = trainer.train(logger=logger)

    json_dir = './json/' + args.dataset + '/'
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    json_path = json_dir + args.model + '-seed' + str(args.seed) + '-' + nowtime + '.json'
    json_result = []
    best_metric = {}
    best_metric['macrof1'] = 0
    best_model_path = None

    json_result.append(metrics)
    if metrics['macrof1'] > best_metric['macrof1']:
        best_metric['macrof1'] = metrics['macrof1']
        best_model_path = model_path

    print("best model path:", best_model_path)
    print("best metric:", best_metric)
    logger.info("best model path:" + best_model_path)
    logger.info("best metric:" + str(best_metric))
    logger.info('--------------------------------------\n')

    with open(json_path, 'w') as file:
        json.dump(json_result, file, indent=4, ensure_ascii=False)
