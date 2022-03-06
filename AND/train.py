#coding:utf-8
from __future__ import division
import torch
import torch.optim as optim
from adjacency import sparse_mx_to_torch_sparse_tensor
from net.gcn_v import GCN_V
import yaml
from easydict import EasyDict
from tensorboardX import SummaryWriter
import numpy as np
import scipy.sparse as sp
import time
import pprint
import sys
import os
import argparse
import math
import pandas as pd
import dgl
import warnings
from tqdm import tqdm


class node_dataset(torch.utils.data.Dataset):
    def __init__(self, node_list, **kwargs):
        self.node_list = node_list

    def __getitem__(self, index):
        return self.node_list[index]

    def __len__(self):
        return len(self.node_list)

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

def adjust_lr(cur_epoch, param, cfg):
    if cur_epoch not in cfg.step_number:
        return
    ind = cfg.step_number.index(cur_epoch)
    for each in optimizer.param_groups:
        each['lr'] = lr

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
    parser.add_argument('--train_Ifile', type=str)
    parser.add_argument('--train_labelfile', type=str)
    parser.add_argument('--test_featfile', type=str)
    parser.add_argument('--test_Ifile', type=str)
    parser.add_argument('--test_labelfile', type=str)
    parser.add_argument('--resume_path', type=str)
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
    print("train hyper parameter list")
    pprint.pprint(cfg)

    # get model
    model = GCN_V(feature_dim=cfg.feat_dim, nhid=cfg.nhid, nclass=cfg.nclass, dropout=0.5)
    model.cuda()

    # get dataset
    scale_max = 80.
    with Timer('load data'):
        train_feature = np.load(cfg.train_featfile)
        train_feature = train_feature / np.linalg.norm(train_feature, axis=1, keepdims=True)
        train_adj = np.load(cfg.train_Ifile)[:, :int(scale_max)]
        train_label_k = np.load(cfg.train_labelfile).astype(np.float32)
        train_label_s = train_label_k / scale_max
        train_feature = torch.FloatTensor(train_feature).cuda()
        train_label_s = torch.FloatTensor(train_label_s).cuda()
        train_data = (train_feature, train_adj, train_label_s)

        test_feature = np.load(cfg.test_featfile)
        test_feature = test_feature / np.linalg.norm(test_feature, axis=1, keepdims=True)
        test_adj = np.load(cfg.test_Ifile)[:, :int(scale_max)]
        test_label_k = np.load(cfg.test_labelfile).astype(np.float32)
        test_label_s = test_label_k / scale_max
        test_feature = torch.FloatTensor(test_feature).cuda()
        test_label_s = torch.FloatTensor(test_label_s).cuda()

        train_dataset = node_dataset(range(len(train_feature)))
        test_dataset  = node_dataset(range(len(test_feature)))
        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=cfg.batchsize,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            drop_last=False)

        test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=cfg.batchsize,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            drop_last=False)

    if cfg.phase == 'train':
        optimizer = optim.SGD(model.parameters(), cfg.lr, momentum=cfg.sgd_momentum, weight_decay=cfg.sgd_weight_decay)
        beg_step = 0
        if cfg.resume_path != None:
            beg_step = int(os.path.splitext(os.path.basename(cfg.resume_path))[0].split('_')[1])
            with Timer('resume model from %s'%cfg.resume_path):
                ckpt = torch.load(cfg.resume_path, map_location='cpu')
                model.load_state_dict(ckpt['state_dict'])

        train_loss_meter = AverageMeter()
        train_kdiff_meter = AverageMeter()
        train_mre_meter = AverageMeter()
        test_loss_meter = AverageMeter()
        test_kdiff_meter = AverageMeter()
        test_mre_meter = AverageMeter()
        writer = SummaryWriter(os.path.join(cfg.outpath), filename_suffix='')

        current_step = beg_step
        break_flag = False
        while 1:
            if break_flag:
                break
            iter_begtime = time.time()
            for _, index in enumerate(train_dataloader):
                if current_step > cfg.total_step:
                    break_flag = True
                    break
                current_step += 1
                cos_lr(current_step, optimizer, cfg)

                batch_feature = train_feature[train_adj[index]]
                batch_label = train_label_s[index]
                batch_k = train_label_k[index]
                batch_data = (batch_feature, batch_label)

                model.train()
                pred_arr, train_loss = model(batch_data, return_loss=True)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                train_loss_meter.update(train_loss.item())
                pred_arr = pred_arr.data.cpu().numpy()

                # add this clip
                k_hat = np.round(pred_arr * scale_max)
                k_hat[np.where(k_hat < 1)[0]] = 1
                k_hat[np.where(k_hat > scale_max)[0]] = scale_max

                train_kdiff = np.abs(k_hat - batch_k)
                train_kdiff_meter.update(train_kdiff.mean())
                train_mre = train_kdiff / batch_k
                train_mre_meter.update(train_mre.mean())
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=current_step)
                writer.add_scalar('loss/train', train_loss.item(), global_step=current_step)
                writer.add_scalar('kdiff/train', train_kdiff_meter.val, global_step=current_step)
                writer.add_scalar('mre/train', train_mre_meter.val, global_step=current_step)
                if current_step % cfg.log_freq == 0:
                    log = "step:{}, step_time:{:.3f}, lr:{:.8f}, trainloss:{:.4f}({:.4f}), trainkdiff:{:.2f}({:.2f}), trainmre:{:.2f}({:.2f}), testloss:{:.4f}({:.4f}), testkdiff:{:.2f}({:.2f}), testmre:{:.2f}({:.2f})".format(current_step, time.time()-iter_begtime, optimizer.param_groups[0]['lr'], train_loss_meter.val, train_loss_meter.avg, train_kdiff_meter.val, train_kdiff_meter.avg, train_mre_meter.val, train_mre_meter.avg, test_loss_meter.val, test_loss_meter.avg, test_kdiff_meter.val, test_kdiff_meter.avg, test_mre_meter.val, test_mre_meter.avg)
                    print(log)
                iter_begtime = time.time()
                if (current_step) % cfg.save_freq == 0 and current_step > 0:
                    torch.save({'state_dict' : model.state_dict(), 'step': current_step}, 
                            os.path.join(cfg.outpath, "ckpt_%s.pth"%(current_step)))
                    
                if (current_step) % cfg.val_freq == 0 and current_step > 0:
                    pred_list = []
                    model.eval()
                    testloss_list = []
                    for step, index in enumerate(tqdm(test_dataloader, desc='test phase', disable=False)):

                        batch_feature = test_feature[test_adj[index]]
                        batch_label = test_label_s[index]
                        batch_data = (batch_feature, batch_label)
            
                        pred, test_loss = model(batch_data, return_loss=True)
                        pred_list.append(pred.data.cpu().numpy())
                        testloss_list.append(test_loss.item())
            
                    pred_list = np.concatenate(pred_list)
                    k_hat, k_arr = pred_list * scale_max, test_label_k

                    # add this clip before eval
                    k_hat = np.round(k_hat)
                    k_hat[np.where(k_hat < 1)[0]] = 1
                    k_hat[np.where(k_hat > scale_max)[0]] = scale_max

                    test_kdiff = np.abs(np.round(k_hat) - k_arr.reshape(-1))
                    test_mre = test_kdiff / k_arr.reshape(-1)
                    test_kdiff_meter.update(test_kdiff.mean())
                    test_mre_meter.update(test_mre.mean())
                    test_loss_meter.update(np.mean(testloss_list))
                    writer.add_scalar('loss/test', test_loss_meter.val, global_step=current_step)
                    writer.add_scalar('kdiff/test', test_kdiff_meter.val, global_step=current_step)
                    writer.add_scalar('mre/test', test_mre_meter.val, global_step=current_step)

        writer.close()
    else:
        ckpt = torch.load(cfg.resume_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])

        pred_list, gcnfeat_list = [], []
        model.eval()
        beg_time = time.time()
        for step, index, in enumerate(test_dataloader):
            batch_feature = test_feature[test_adj[index]]
            batch_label = test_label_s[index]
            batch_data = (batch_feature, batch_label)

            pred, gcnfeat = model(batch_data, output_feat=True)
            pred_list.append(pred.data.cpu().numpy())
            gcnfeat_list.append(gcnfeat.data.cpu().numpy())
        print("time use %.4f"%(time.time()-beg_time))

        pred_list = np.concatenate(pred_list)
        gcnfeat_arr = np.vstack(gcnfeat_list)
        gcnfeat_arr = gcnfeat_arr / np.linalg.norm(gcnfeat_arr, axis=1, keepdims=True)
        tag = os.path.splitext(os.path.basename(cfg.resume_path))[0]

        print("stat")
        k_hat, k_arr = pred_list * scale_max, test_label_k

        # add this clip before eval
        k_hat = np.round(k_hat)
        k_hat[np.where(k_hat < 1)[0]] = 1
        k_hat[np.where(k_hat > scale_max)[0]] = scale_max
        np.save(os.path.join(cfg.outpath, 'k_infer_pred'), np.round(k_hat))

    print("time use", time.time() - beg_time)
