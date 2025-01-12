import os
import tqdm
import torch.cuda

from utils.utils import data2gpu, data2gpu_noemo, Averager, twitter_metrics, weibo_metrics, Recorder
from utils.dataloader import get_twitter_dataloader, get_weibo_dataloader

from models.bertemo import BERTEmoModel
from models.cbert import cBERTModel
from models.defend import dEFENDBERTModel
from models.casfend import CASFENDModel
from models.kahan import KAHANModel
from models.cameredit import CameReditModel


class Trainer():
    def __init__(self, args):
        self.args = args

        self.save_path = os.path.join(self.args.save_param_dir, self.args.model)
        if os.path.exists(self.save_path):
            self.save_param_dir = self.save_path
        else:
            self.save_param_dir = os.makedirs(self.save_path)

    def train(self, logger):
        if logger:
            logger.info('start training......')
        save_path = os.path.join(self.save_path, 'parameter_bert-' + self.args.dataset + '.pkl')

        # build models
        category = 4 if 'twitter' in self.args.dataset else 2
        if self.args.model == 'bertemo':
            self.model = BERTEmoModel(self.args.pretrain_name, self.args.emb_dim, 270 if 'twitter' in self.args.dataset else 275,
                                      self.args.inner_dim, self.args.dropout, category)
        elif self.args.model == 'cbert':
            self.model = cBERTModel(self.args.pretrain_name, self.args.emb_dim, self.args.inner_dim, self.args.dropout, category)
        elif self.args.model == 'defend':
            self.model = dEFENDBERTModel(self.args.pretrain_name, self.args.emb_dim, self.args.inner_dim, self.args.dropout, category)
        elif self.args.model == 'casfend':
            self.model = CASFENDModel(self.args.pretrain_name, self.args.emb_dim, 270 if 'twitter' in self.args.dataset else 275,
                                      self.args.inner_dim, self.args.dropout, category)
        elif self.args.model == 'kahan':
            self.model = KAHANModel(self.args.pretrain_name, self.args.emb_dim, self.args.inner_dim, self.args.dropout, category)
        else:
            self.model = CameReditModel(self.args.pretrain_name, self.args.emb_dim, self.args.inner_dim, self.args.dropout, category)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        recorder = Recorder(self.args.early_stop, logger)

        if 'twitter' in self.args.dataset:
            dataloader = get_twitter_dataloader
        else:
            dataloader = get_weibo_dataloader

        train_loader = dataloader(self.args.data_path + 'train.json', self.args.data_path + 'emotions/train.npy',
                                  self.args.data_path + 'entity_train.json', self.args.data_path + 'generated/generated_train.json',
                                  self.args.max_len, self.args.train_k, self.args.train_m, self.args.batch_size,
                                  shuffle=True, pretrain_name=self.args.pretrain_name)

        val_loader = dataloader(self.args.data_path + 'val.json', self.args.data_path + 'emotions/val.npy',
                                self.args.data_path + 'entity_val.json', self.args.data_path + 'generated/generated_val.json',
                                self.args.max_len, self.args.test_k, self.args.test_m, self.args.batch_size,
                                shuffle=True, pretrain_name=self.args.pretrain_name)

        test_loader = dataloader(self.args.data_path + 'test.json', self.args.data_path + 'emotions/test.npy',
                                 self.args.data_path + 'entity_test.json', self.args.data_path + 'generated/generated_test.json',
                                 self.args.max_len, self.args.test_k, self.args.test_m, self.args.batch_size,
                                 shuffle=True, pretrain_name=self.args.pretrain_name)

        loss_fn = torch.nn.CrossEntropyLoss()

        diff_part = ["bertModel.embeddings", "bertModel.encoder"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in diff_part)],
                "weight_decay": 0.0,
                "lr": self.args.lr
            },
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in diff_part)],
                "weight_decay": 0.0,
                "lr": self.args.mlp_lr
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=self.args.adam_epsilon)

        logger.info("Training the fake news detector based on {}".format(self.args.pretrain_name))
        for epoch in range(self.args.epoch):
            self.model.train()
            train_data_iter = tqdm.tqdm(train_loader)
            avg_loss = Averager()

            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, use_cuda=True)
                label = batch_data['label']

                pred = self.model(**batch_data)
                loss = loss_fn(pred, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss.add(loss.item())
            logger.info(10 * '-' + '\nTraining Epoch {}; Loss {};\n'.format(epoch + 1, avg_loss.item()) + 10 * '-')

            results = self.test(val_loader)
            mark = recorder.add(results)  # early stop with validation metrics
            if mark == 'save':
                torch.save(self.model.state_dict(), save_path)
            elif mark == 'esc':
                break
            else:
                continue

        logger.info("Stage: testing...")
        self.model.load_state_dict(torch.load(save_path))

        future_results = self.test(test_loader)

        logger.info("test score: {}.".format(future_results))
        logger.info("lr: {}, aug_prob: {}, avg test score: {}.\n\n".format(self.args.lr, self.args.aug_prob, future_results['macrof1']))
        print('test results:', future_results)

        return future_results, save_path

    def test(self, dataloader):
        pred = []
        label = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)

        with torch.no_grad():
            for step_n, batch in enumerate(data_iter):
                batch_data = data2gpu(batch, use_cuda=True)
                batch_label = batch_data['label']

                batch_pred = self.model(**batch_data)
                batch_pred = torch.nn.functional.softmax(batch_pred)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_pred.detach().cpu().numpy().tolist())

        return twitter_metrics(label, pred) if 'twitter' in self.args.dataset else weibo_metrics(label, pred)
