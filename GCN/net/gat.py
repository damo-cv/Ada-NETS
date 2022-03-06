#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from dgl.nn.pytorch import SAGEConv
from .optim_modules import BallClusterLearningLoss, ClusterLoss
import torch.nn.functional as F

class GCN_V(nn.Module):
    def __init__(self, feature_dim, nhid, nclass, dropout=0, losstype='allall', margin=1., pweight=4., pmargin=1.0):
        super(GCN_V, self).__init__()

        self.sage1 = SAGEConv(feature_dim, nhid, aggregator_type='gcn', activation=F.relu)
        self.sage2 = SAGEConv(nhid, nhid, aggregator_type='gcn', activation=F.relu)

        self.nclass = nclass
        self.fc = nn.Sequential(nn.Linear(nhid, nhid), nn.PReLU(nhid))
        self.loss = torch.nn.MSELoss()
        self.bclloss = ClusterLoss(losstype=losstype, margin=margin, alpha_pos=pweight, pmargin=pmargin)

    def forward(self, data, output_feat=False, return_loss=False):
        assert not output_feat or not return_loss
        x, block_list, label, idlabel = data[0], data[1], data[2], data[3]

        # layer1
        gcnfeat = self.sage1(block_list[0], x)
        gcnfeat = F.normalize(gcnfeat, p=2, dim=1)

        # layer2
        gcnfeat = self.sage2(block_list[1], gcnfeat)

        # layer3
        fcfeat = self.fc(gcnfeat)
        fcfeat = F.normalize(fcfeat, dim=1)

        if output_feat:
            return fcfeat, gcnfeat

        if return_loss:
            bclloss_dict = self.bclloss(fcfeat, label)
            return bclloss_dict

        return fcfeat
