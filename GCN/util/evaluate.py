#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
import argparse
import numpy as np
import util.metrics as metrics
import time

class TextColors:
    #HEADER = '\033[35m'
    #OKBLUE = '\033[34m'
    #OKGREEN = '\033[32m'
    #WARNING = '\033[33m'
    #FATAL = '\033[31m'
    #ENDC = '\033[0m'
    #BOLD = '\033[1m'
    #UNDERLINE = '\033[4m'
    HEADER = ''
    OKBLUE = ''
    OKGREEN = ''
    WARNING = ''
    FATAL = ''
    ENDC = ''
    BOLD = ''
    UNDERLINE = ''

class Timer():
    def __init__(self, name='task', verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            print('[Time] {} consumes {:.4f} s'.format(
                self.name,
                time.time() - self.start))
        return exc_type is None


def _read_meta(fn):
    labels = list()
    lb_set = set()
    with open(fn) as f:
        for lb in f.readlines():
            lb = int(lb.strip())
            labels.append(lb)
            lb_set.add(lb)
    return np.array(labels), lb_set


def evaluate(gt_labels, pred_labels, metric='pairwise'):
    if isinstance(gt_labels, str) and isinstance(pred_labels, str):
        print('[gt_labels] {}'.format(gt_labels))
        print('[pred_labels] {}'.format(pred_labels))
        gt_labels, gt_lb_set = _read_meta(gt_labels)
        pred_labels, pred_lb_set = _read_meta(pred_labels)

        print('#inst: gt({}) vs pred({})'.format(len(gt_labels),
                                                 len(pred_labels)))
        print('#cls: gt({}) vs pred({})'.format(len(gt_lb_set),
                                                len(pred_lb_set)))

    metric_func = metrics.__dict__[metric]

    with Timer('evaluate with {}{}{}'.format(TextColors.FATAL, metric,
                                             TextColors.ENDC), verbose=False):
        result = metric_func(gt_labels, pred_labels)
    if isinstance(result, np.float):
        #print('{}{}: {:.4f}{}'.format(TextColors.OKGREEN, metric, result, TextColors.ENDC))
        res_str = '{}{}: {:.4f}{}'.format(TextColors.OKGREEN, metric, result, TextColors.ENDC)
    else:
        from collections import Counter
        singleton_num = len( list( filter(lambda x: x==1, Counter(pred_labels).values()) ) )
        ave_pre, ave_rec, fscore = result
        #print('{}ave_pre: {:.4f}, ave_rec: {:.4f}, fscore: {:.4f}{}, cluster_num: {}, singleton_num: {}'.format(
        #    TextColors.OKGREEN, ave_pre, ave_rec, fscore, TextColors.ENDC,  len(np.unique(pred_labels)), singleton_num))
        res_str = '{}{}: ave_pre: {:.4f}, ave_rec: {:.4f}, fscore: {:.4f}{}, cluster_num: {}, singleton_num: {}'.format(
                TextColors.OKGREEN, metric, ave_pre, ave_rec, fscore, TextColors.ENDC,  len(np.unique(pred_labels)), singleton_num)
    #return ave_pre, ave_rec, fscore
    return res_str


if __name__ == '__main__':
    metric_funcs = inspect.getmembers(metrics, inspect.isfunction)
    metric_names = [n for n, _ in metric_funcs]

    parser = argparse.ArgumentParser(description='Evaluate Cluster')
    parser.add_argument('--gt_labels', type=str, required=True)
    parser.add_argument('--pred_labels', type=str, required=True)
    parser.add_argument('--metric', default='pairwise', choices=metric_names)
    args = parser.parse_args()

    evaluate(args.gt_labels, args.pred_labels, args.metric)
