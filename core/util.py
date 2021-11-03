"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util
@Time: 4/5/19 3:47 PM
"""


import numpy as np
import torch
import torch.nn.functional as F
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
