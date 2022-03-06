#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
#from .utils import GraphConv, MeanAggregator
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
import math


class huberloss(nn.Module):
    def __init__(self, delta):
        super(huberloss, self).__init__()
        self.delta = delta

    def forward(self, input_arr, target_arr):
        rate = input_arr / target_arr - 1
        loss = torch.where(torch.abs(rate) <= self.delta, 0.5*rate*rate, (torch.abs(rate) - 0.5*self.delta) * self.delta)
        return loss.mean()

class MREloss(nn.Module):
    def __init__(self):
        super(MREloss, self).__init__()

    def forward(self, input_arr, target_arr):
        loss = torch.abs(input_arr / target_arr - 1)
        return loss.mean()


class GCN_V(nn.Module):
    def __init__(self, feature_dim, nhid, nclass, dropout=0):
        super(GCN_V, self).__init__()
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=feature_dim, num_layers=1, batch_first=True, dropout=dropout, bidirectional=True)
        self.out_proj = nn.Linear(2*feature_dim, feature_dim, bias=True)

        self.nclass = nclass
        self.mlp = nn.Sequential(
                nn.Linear(feature_dim, nhid), nn.PReLU(nhid), nn.Dropout(p=dropout),
                nn.Linear(nhid, feature_dim), nn.PReLU(feature_dim), nn.Dropout(p=dropout),
                )
        self.regressor = nn.Linear(feature_dim, 1)
        #self.loss = torch.nn.MSELoss()
        #self.loss = MREloss()
        self.loss = huberloss(delta=1.0)

    def forward(self, data, output_feat=False, return_loss=False):
        assert not output_feat or not return_loss
        batch_feat, batch_label = data[0], data[1]

        # lstm block
        out, (hn, cn) = self.lstm(batch_feat)
        out = self.out_proj(out)
        out = (out + batch_feat) / math.sqrt(2.)

        # normalize before mean
        out = F.normalize(out, 2, dim=-1)
        out = out.mean(dim=1)
        out = F.normalize(out, 2, dim=-1)

        # mlp block
        residual = out
        out = self.mlp(out)
        out = (residual + out ) / math.sqrt(2.)

        # regressor block
        pred = self.regressor(out).view(-1)

        if output_feat:
            return pred, residual

        if return_loss:
            loss = self.loss(pred, batch_label)
            return pred, loss

        return pred


def gcn_v(feature_dim, nhid, nclass=1, dropout=0., **kwargs):
    model = GCN_V(feature_dim=feature_dim,
                  nhid=nhid,
                  nclass=nclass,
                  dropout=dropout)
    return model
