from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score, log_loss
import numpy as np


class Recorder():

    def __init__(self, early_step, logger):
        self.logger = logger
        self.max = {'macrof1': 0}
        self.cur = {'macrof1': 0}
        self.maxindex = 0
        self.curindex = 0
        self.early_step = early_step

    def add(self, x):
        self.cur = x
        self.curindex += 1
        self.logger.info("curent" + str(self.cur))
        return self.judge()

    def judge(self):
        if self.cur['macrof1'] > self.max['macrof1']:
            self.max = self.cur
            self.maxindex = self.curindex
            self.showfinal()
            return 'save'
        self.showfinal()
        if self.curindex - self.maxindex >= self.early_step:
            return 'esc'
        else:
            return 'continue'

    def showfinal(self):
        self.logger.info("Max" + str(self.max))


def twitter_metrics(y_true, y_pred):
    all_metrics = {}

    all_metrics['auc'] = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr')
    all_metrics['weightauc'] = roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr')
    # all_metrics['spauc'] = roc_auc_score(y_true, y_pred, average='macro', max_fpr=0.1)
    predicted_classes = np.argmax(np.array(y_pred), axis=1)
    # y_pred = np.around(np.array(y_pred)).astype(int)
    all_metrics['macrof1'] = f1_score(y_true, predicted_classes, average='macro')
    all_metrics['microf1'] = f1_score(y_true, predicted_classes, average='micro')
    all_metrics['acc'] = accuracy_score(y_true, predicted_classes)
    all_metrics['recall'] = recall_score(y_true, predicted_classes, average='macro')
    all_metrics['precision'] = precision_score(y_true, predicted_classes, average='macro')
    all_metrics['precision'] = precision_score(y_true, predicted_classes, average='macro')
    all_metrics['logloss'] = log_loss(y_true, y_pred)
    # all_metrics['f1_real'], all_metrics['f1_fake'] = f1_score(y_true, y_pred, average=None)
    # all_metrics['recall_real'], all_metrics['recall_fake'] = recall_score(y_true, y_pred, average=None)
    # all_metrics['precision_real'], all_metrics['precision_fake'] = precision_score(y_true, y_pred, average=None)
    
    return all_metrics


def weibo_metrics(y_true, y_pred):
    all_metrics = {}

    y_pred = np.argmax(np.array(y_pred), axis=1).tolist()
    all_metrics['auc'] = roc_auc_score(y_true, y_pred, average='macro')
    all_metrics['spauc'] = roc_auc_score(y_true, y_pred, average='macro', max_fpr=0.1)
    y_pred = np.around(np.array(y_pred)).astype(int).tolist()
    all_metrics['macrof1'] = f1_score(y_true, y_pred, average='macro')  # ‘metric’ is macro f1 that is the best important metric
    all_metrics['f1_real'], all_metrics['f1_fake'] = f1_score(y_true, y_pred, average=None)
    all_metrics['recall'] = recall_score(y_true, y_pred, average='macro')
    all_metrics['recall_real'], all_metrics['recall_fake'] = recall_score(y_true, y_pred, average=None)
    all_metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    all_metrics['precision_real'], all_metrics['precision_fake'] = precision_score(y_true, y_pred, average=None)
    all_metrics['acc'] = accuracy_score(y_true, y_pred)

    return all_metrics

def data2gpu(batch, use_cuda):
    if use_cuda:
        batch_data = {
            'content': batch[0].cuda(),
            'content_masks': batch[1].cuda(),
            'entity': batch[2].cuda(),
            'comment': batch[3].cuda(),
            'comment_num': batch[4].cuda(),
            'emotion': batch[5].cuda(),
            'label': batch[6].cuda(),
        }
    else:
        batch_data = {
            'content': batch[0],
            'content_masks': batch[1],
            'entity': batch[2],
            'comment': batch[3],
            'comment_num': batch[4],
            'emotion': batch[5],
            'label': batch[6],
        }
    return batch_data


def data2gpu_noemo(batch, use_cuda):
    if use_cuda:
        batch_data = {
            'content': batch[0].cuda(),
            'content_masks': batch[1].cuda(),
            'label': batch[2].cuda(),
        }
    else:
        batch_data = {
            'content': batch[0],
            'content_masks': batch[1],
            'label': batch[2],
        }
    return batch_data

class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v