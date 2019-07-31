#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : performance
# @Date     : 07/21/2019 13:50:55
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :

from __future__ import print_function

import os
import pickle

import numpy as np


def calculate_performance(id):
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions/exp_1/case_{}/Output_{}'.format(id, id)
    fn = '01DenseSubgraphNodes.txt'
    subgraph = set()
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            node = int(line.strip())
            subgraph.add(node)

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions/exp_1/case_{}'.format(id)
    fn = 'attributes_biased_{}.pkl'.format(id)
    with open(os.path.join(path, fn), 'rb') as rfile:
        dataset = pickle.load(rfile)

    true_subgraph = dataset['subgraph']
    prec, rec, fm, iou = evaluate(true_subgraph, subgraph)

    print(prec, rec, fm, iou)




def evaluate(true_subgraph, pred_subgraph):
    true_subgraph, pred_subgraph = set(true_subgraph), set(pred_subgraph)

    prec = rec = fm = iou = 0.

    intersection = true_subgraph & pred_subgraph
    union = true_subgraph | pred_subgraph

    if not 0. == len(pred_subgraph):
        prec = len(intersection) / float(len(pred_subgraph))

    if not 0. == len(true_subgraph):
        rec = len(intersection) / float(len(true_subgraph))

    if prec + rec > 0.:
        fm = (2. * prec * rec) / (prec + rec)

    if union:
        iou = len(intersection) / float(len(union))

    return prec, rec, fm, iou


if __name__ == '__main__':

    for id in range(10):
        calculate_performance(id)