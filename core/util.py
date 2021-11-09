"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util
@Time: 4/5/19 3:47 PM
"""


import numpy as np
import torch
import torch.nn.functional as F
import sklearn.metrics as metrics
from evaluation import *
# import torch.nn.BCELoss
# import torch.nn.BCEWithLogitsLoss


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    # gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        
        one_hot = gold
        n_class = pred.size(1)
        n_lab = torch.sum(one_hot,dim=1)

        # one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = (one_hot * (1 - eps/(n_lab[:, None]))) + \
            ((1 - one_hot) * eps / (n_class - n_lab)[:, None])
        
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.multilabel_soft_margin_loss(pred, gold)

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

        
def evaluate(true, pred_prob, icvec, nth=10):
    test_eval_metrics = {}
    
    pred = np.zeros(pred_prob.shape, dtype=bool)
    pred[np.arange(len(pred_prob)), pred_prob.argmax(axis=1)] = 1
    pred[(pred_prob > 0.5)] = 1
    pred = pred.astype(float)
    
    test_eval_metrics['acc'] = metrics.accuracy_score(true, (pred_prob > 0.5).astype(float))
    # https://github.com/stamakro/GCN-for-Structure-and-Function/blob/fb148d5579adbb805c1d054d24216db285198540/scripts/model.py#L109-L142
    # Average precision score
    test_eval_metrics['avg_avgprec'] = average_precision_score(true, pred_prob, average='samples')
    # ROC AUC score
    ii = np.where(np.sum(true, 0) > 0)[0]
    test_eval_metrics['avg_rocauc'] = roc_auc_score(true[:, ii], pred_prob[:, ii], average='macro')
    # Minimum semantic distance
    test_eval_metrics['avg_sdmin'] = smin(true, pred_prob, icvec, nrThresholds=nth)
    # Maximum F-score
    nth=10
    test_eval_metrics['avg_fmax'] = fmax(true, pred_prob, nrThresholds=nth)
    return test_eval_metrics