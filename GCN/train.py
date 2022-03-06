#coding:utf-8
from __future__ import division
import torch
import torch.optim as optim
from adjacency import sparse_mx_to_torch_sparse_tensor
from net.gat import GCN_V
#from net.gcn_v import GCN_V
#from net.softmaxloss import GCN_V
import yaml
from easydict import EasyDict
from tensorboardX import SummaryWriter
import numpy as np
import scipy.sparse as sp
import time
import sys
import os
import apex
from apex import amp
import dgl
import math
import argparse
import pprint
from abc import ABC, abstractproperty, abstractmethod
from collections.abc import Mapping

class Collator(ABC):
    @abstractproperty
    def dataset(self):
        raise NotImplementedError

    @abstractmethod
    def collate(self, items):
        raise NotImplementedError

class multigraph_NodeCollator(Collator):
    def __init__(self, order_graph, ngbr_graph, nids, block_sampler):
        self.order_graph = order_graph
        self.ngbr_graph = ngbr_graph
        self.nids = nids
        self.block_sampler = block_sampler
        self._dataset = nids

    @property
    def dataset(self):
        return self._dataset

    def collate(self, items):
        # use collate to fasten
        #blocks = self.block_sampler.sample_blocks(self.g, items)

        seed_node0 = items
        frontier0 = dgl.sampling.sample_neighbors(self.order_graph, seed_node0, 128, replace=False)  # sample 128 from 256
        block0 = dgl.to_block(frontier0, seed_node0)

        seed_node1 = {ntype: block0.srcnodes[ntype].data[dgl.NID] for ntype in block0.srctypes}
        frontier1 = dgl.sampling.sample_neighbors(self.ngbr_graph, seed_node1, 80, replace=False)
        block1 = dgl.to_block(frontier1, seed_node1)
        block1.create_format_()

        seed_node2 = {ntype: block1.srcnodes[ntype].data[dgl.NID] for ntype in block1.srctypes}
        frontier2 = dgl.sampling.sample_neighbors(self.ngbr_graph, seed_node2, 80, replace=False)
        block2 = dgl.to_block(frontier2, seed_node2)
        block2.create_format_()

        blocks = [block2, block1]
        input_nodes = blocks[0].srcdata[dgl.NID]
        output_nodes = blocks[-1].dstdata[dgl.NID]
        return input_nodes, output_nodes, blocks

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / (self.count + 1e-10)

def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    # if rowsum <= 0, keep its previous value
    rowsum[rowsum <= 0] = 1
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

class Timer():
    def __init__(self, name='task', verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        print('[begin {}]'.format(self.name))
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            print('[done {}] use {:.3f} s'.format(self.name, time.time() - self.start))
        return exc_type is None

def adjust_lr(cur_epoch, optimizer, cfg):
    if cur_epoch not in cfg.step_number:
        return
    ind = cfg.step_number.index(cur_epoch)
    for each in optimizer.param_groups:
        each['lr'] = cfg.lr * cfg.factor ** (ind+1)

def cos_lr(current_step, optimizer, cfg):
    if current_step < cfg.warmup_step:
        rate = 1.0 * current_step / cfg.warmup_step
        lr = cfg.lr * rate
    else:
        n1 = cfg.total_step - cfg.warmup_step
        n2 = current_step - cfg.warmup_step
        rate = (1 + math.cos(math.pi * n2 / n1)) / 2
        lr = cfg.lr * rate
    for each in optimizer.param_groups:
        each['lr'] = lr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--outpath', type=str)
    parser.add_argument('--phase', type=str)
    parser.add_argument('--train_featfile', type=str)
    parser.add_argument('--train_adjfile', type=str)
    parser.add_argument('--train_orderadjfile', type=str)
    parser.add_argument('--train_labelfile', type=str)
    parser.add_argument('--test_featfile', type=str)
    parser.add_argument('--test_adjfile', type=str)
    parser.add_argument('--test_labelfile', type=str)
    parser.add_argument('--resume_path', type=str)
    parser.add_argument('--losstype', type=str)
    parser.add_argument('--margin', type=float)
    parser.add_argument('--pweight', type=float)
    parser.add_argument('--pmargin', type=float)
    parser.add_argument('--topk', type=int)
    args = parser.parse_args()

    beg_time = time.time()
    config = yaml.load(open(args.config_file, "r"), Loader=yaml.FullLoader)
    cfg = EasyDict(config)
    cfg.step_number = [int(r * cfg.total_step) for r in cfg.lr_step]

    # force assignment
    for key, value in args._get_kwargs():
        cfg[key] = value
    #cfg[list(dict(train_adjfile=train_adjfile).keys())[0]] = train_adjfile
    #cfg[list(dict(train_labelfile=train_labelfile).keys())[0]] = train_labelfile
    #cfg[list(dict(test_adjfile=test_adjfile).keys())[0]] = test_adjfile
    #cfg[list(dict(test_labelfile=test_labelfile).keys())[0]] = test_labelfile
    cfg.var = EasyDict()
    print("train hyper parameter list")
    pprint.pprint(cfg)


    # get model
    model = GCN_V(feature_dim=cfg.feat_dim, nhid=cfg.nhid, nclass=cfg.nclass, dropout=0., losstype=cfg.losstype, margin=cfg.margin,
            pweight=cfg.pweight, pmargin=cfg.pmargin)

    # get dataset
    with Timer('load data'):
        if cfg.phase == 'train':
            featfile, adjfile, labelfile = cfg.train_featfile, cfg.train_adjfile, cfg.train_labelfile
            order_adj = sp.load_npz(cfg.train_orderadjfile).astype(np.float32)
            order_graph = dgl.from_scipy(order_adj)
        else:
            featfile, adjfile, labelfile = cfg.test_featfile, cfg.test_adjfile, cfg.test_labelfile
        features = np.load(featfile)
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        adj = sp.load_npz(adjfile).astype(np.float32)
        graph = dgl.from_scipy(adj)
        label_arr = np.load(labelfile)
        features = torch.FloatTensor(features)
        #adj = sparse_mx_to_torch_sparse_tensor(adj)
        label_cpu = torch.LongTensor(label_arr)
        if cfg.cuda:
            model.cuda()
            features = features.cuda()
            #adj = adj.cuda()
            labels = label_cpu.cuda()
        #data = (features, adj, labels)

    # get train
    if cfg.phase == 'train':
        # get optimizer
        pretrain_pool = True
        pretrain_pool = False
        if pretrain_pool:
            pool_weight, net_weight = [], []
            for k, v in model.named_parameters():
                if 'pool.' in k:
                    pool_weight += [v]
                else:
                    net_weight += [v]
            param_list = [{'params': pool_weight}, {'params': net_weight, 'lr': 0.}]
            optimizer = optim.SGD(param_list, cfg.lr, momentum=cfg.sgd_momentum, weight_decay=cfg.sgd_weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(), cfg.lr, momentum=cfg.sgd_momentum, weight_decay=cfg.sgd_weight_decay)

        if cfg.fp16:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1", keep_batchnorm_fp32=None, loss_scale='dynamic')

        beg_step = 0
        if cfg.resume_path != None:
            beg_step = int(os.path.splitext(os.path.basename(cfg.resume_path))[0].split('_')[1])
            with Timer('resume model from %s'%cfg.resume_path):
                ckpt = torch.load(cfg.resume_path, map_location='cpu')
                model.load_state_dict(ckpt['state_dict'])

        totalloss_meter = AverageMeter()
        bclloss_pos_meter = AverageMeter()
        bclloss_neg_meter = AverageMeter()
        keeploss_meter = AverageMeter()
        before_edge_num_meter = AverageMeter()
        after_edge_num_meter = AverageMeter()
        acc_meter = AverageMeter()
        prec_meter = AverageMeter()
        recall_meter = AverageMeter()
        leftprec_meter = AverageMeter()
        writer = SummaryWriter(os.path.join(cfg.outpath), filename_suffix='')
        #sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        #sampler = dgl.dataloading.MultiLayerNeighborSampler([cfg.topk, cfg.topk, 128])
        #sampler = dgl.dataloading.MultiLayerNeighborSampler([None, None, 128])
        #dataloader = dgl.dataloading.NodeDataLoader(
        #    order_graph,
        #    np.arange(order_graph.number_of_nodes()),
        #    sampler,
        #    batch_size=cfg.batchsize,
        #    shuffle=True,
        #    drop_last=False,
        #    num_workers=4)
        #sampler = dgl.dataloading.MultiLayerNeighborSampler([128])
        sampler = None
        collator = multigraph_NodeCollator(order_graph, graph, np.arange(len(features)), sampler)  # v4
        dataloader = torch.utils.data.DataLoader(
            dataset=collator.dataset,
            batch_size=cfg.batchsize,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
            collate_fn=collator.collate,
            )

        current_step = beg_step
        break_flag = False
        while 1:
            #adjust_lr(current_step, optimizer.param_groups, cfg)
            if break_flag:
                break
            for _, (src_idx, dst_idx, blocks) in enumerate(dataloader):
                if current_step > cfg.total_step:
                    break_flag = True
                    break
                iter_begtime = time.time()
                current_step += 1
                cos_lr(current_step, optimizer, cfg)
                #src_idx = blocks[0].srcdata[dgl.NID].numpy()
                #dst_idx = blocks[-1].dstdata[dgl.NID].numpy()

                batch_feature = features[src_idx, :]
                #batch_adj = sparse_mx_to_torch_sparse_tensor(adj[src_idx, :][:, src_idx]).cuda()
                #batch_adj = torch.from_numpy( row_normalize(adj[src_idx, :][:, src_idx]).todense() ).cuda()
                batch_block = [block.to(0) for block in blocks] # need not row normalize, because the attention weight edge
                batch_label = labels[dst_idx]
                batch_idlabel = labels[src_idx]
                batch_data = (batch_feature, batch_block, batch_label, batch_idlabel)
                bclloss_dict  = model(batch_data, return_loss=True)
                loss = bclloss_dict['ctrd_pos'] + bclloss_dict['ctrd_neg']

                optimizer.zero_grad()
                if cfg.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()

                totalloss_meter.update(loss.item())
                bclloss_pos_meter.update(bclloss_dict['ctrd_pos'].item())
                bclloss_neg_meter.update(bclloss_dict['ctrd_neg'].item())

                writer.add_scalar('loss/total', loss.item(), global_step=current_step)
                writer.add_scalar('loss/bcl_pos', bclloss_dict['ctrd_pos'].item(), global_step=current_step)
                writer.add_scalar('loss/bcl_neg', bclloss_dict['ctrd_neg'].item(), global_step=current_step)
                if current_step % cfg.log_freq == 0:
                    log = "step{}/{}, iter_time:{:.3f}, lr:{:.4f}, loss:{:.4f}({:.4f}), bclloss_pos:{:.8f}({:.8f}), bclloss_neg:{:.4f}({:.4f}) ".format(current_step, cfg.total_step, time.time()-iter_begtime, optimizer.param_groups[0]['lr'], totalloss_meter.val, totalloss_meter.avg, bclloss_pos_meter.val, bclloss_pos_meter.avg, bclloss_neg_meter.val, bclloss_neg_meter.avg)
                    print(log)
                if (current_step+1) % cfg.save_freq == 0 and current_step > 0:
                    torch.save({'state_dict' : model.state_dict(), 'step': current_step+1}, 
                            os.path.join(cfg.outpath, "ckpt_%s.pth"%(current_step+1)))
        writer.close()
    else:
        with Timer('resume model from %s'%cfg.resume_path):
            ckpt = torch.load(cfg.resume_path, map_location='cpu')
            model.load_state_dict(ckpt['state_dict'])
            model.eval()

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.NodeDataLoader(
            graph,
            np.arange(graph.number_of_nodes()),
            sampler,
            batch_size=1024,
            shuffle=False,
            drop_last=False,
            num_workers=16)

        gcnfeat_list, fcfeat_list = [], []
        leftprec_meter = AverageMeter()
        beg_time = time.time()
        for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            src_idx = blocks[0].srcdata[dgl.NID].numpy()
            dst_idx = blocks[-1].dstdata[dgl.NID].numpy()
            #zip(block.srcnodes(), block.srcdata[dgl.NID])
            #zip(block.dstnodes(), block.dstdata[dgl.NID])
            batch_feature = features[src_idx, :]
            #batch_adj = sparse_mx_to_torch_sparse_tensor(adj[src_idx, :][:, src_idx]).cuda()

            #batch_adj = torch.from_numpy(adj[src_idx, :][:, src_idx].todense()).cuda() # no sample and no row normalize again
            batch_block = [block.to(0) for block in blocks]
            batch_label = labels[dst_idx]
            batch_idlabel = labels[src_idx]
            batch_data = (batch_feature, batch_block, batch_label, batch_idlabel)
            #fcfeat, gcnfeat, before_edge_num, after_edge_num, acc_rate, prec, recall, left_prec = model(batch_data, output_feat=True)
            fcfeat, gcnfeat = model(batch_data, output_feat=True)

            fcfeat_list.append(fcfeat.data.cpu().numpy())
            gcnfeat_list.append(gcnfeat.data.cpu().numpy())
            #leftprec_meter.update(left_prec)
            #if step % 1 == 0:
            #    log = "step %s/%s"%(step, len(dataloader))
            #    print(log)
        print("time use %.4f"%(time.time()-beg_time))

        fcfeat_arr = np.vstack(fcfeat_list)
        gcnfeat_arr = np.vstack(gcnfeat_list)
        fcfeat_arr = fcfeat_arr / np.linalg.norm(fcfeat_arr, axis=1, keepdims=True)
        gcnfeat_arr = gcnfeat_arr / np.linalg.norm(gcnfeat_arr, axis=1, keepdims=True)
        tag = os.path.splitext(os.path.basename(cfg.resume_path))[0]
        np.save(os.path.join(cfg.outpath, 'fcfeat_%s'%tag), fcfeat_arr)
        np.save(os.path.join(cfg.outpath, 'gcnfeat_%s'%tag), gcnfeat_arr)

    print("time use", time.time() - beg_time)
