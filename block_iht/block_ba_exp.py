#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : block_ba_exp
# @Date     : 07/21/2019 17:54:11
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :

from __future__ import print_function

import os
import sys

sys.path.append(os.path.abspath(''))

import logging
from logging.config import fileConfig

fileConfig('../logging.conf')
# note: logger is not thread-safe, pls be careful when running multi-threads with logging
logger = logging.getLogger('fei')

import time
import pickle
import multiprocessing

import numpy as np
import networkx as nx

from impl_block_3 import optimize
from utils import evaluate_block
from utils import post_process_block

DATASET = 'ba'

def run_dataset(paras):
    dataset, sparsity, trade_off, learning_rate, max_iter, epsilon, write_to_dir, num_nodes, deg, normalize = paras

    if not write_to_dir:
        logger = logging.getLogger('fei')
    else:
        log_fn = '{}_nodes_{}_deg_{}_sparsity_{:d}_trade_{}_lr_{}_{}.txt'.format(DATASET, num_nodes, deg, sparsity, trade_off, learning_rate, normalize)

        if os.path.isfile(os.path.join(write_to_dir, log_fn)):
            print('file exist !!!')
            return

        logger = logging.getLogger(log_fn)
        formatter = logging.Formatter('')
        file_handler = logging.FileHandler(filename=os.path.join(write_to_dir, log_fn), mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    all_performance = []
    logger.debug('-' * 5 + ' setting ' + '-' * 5)
    logger.debug('sparsity: {:d}'.format(sparsity))
    logger.debug('learning rate: {:.5f}'.format(learning_rate))
    logger.debug('trade off: {:.5f}'.format(trade_off))
    for i, instance in enumerate(dataset):
        logger.debug('instance: {:d}'.format(i))

        if normalize:
            logger.debug('feature normalized')
            instance['features'] = instance['features'] / np.max(instance['features'])

        opt_x, run_time = optimize(instance, sparsity, trade_off, learning_rate, max_iter, epsilon, logger=None)

        raw_pred_subgraph = np.nonzero(opt_x)[0]

        prec, rec, fm, iou = evaluate_block(instance['true_subgraph'], raw_pred_subgraph)

        logger.debug('-' * 5 + ' performance of raw prediction ' + '-' * 5)
        logger.debug('precision: {:.5f}'.format(prec))
        logger.debug('recall   : {:.5f}'.format(rec))
        logger.debug('f-measure: {:.5f}'.format(fm))
        logger.debug('iou      : {:.5f}'.format(iou))

        refined_pred_subgraph = post_process_block(instance['graph'], raw_pred_subgraph, dataset=DATASET)
        refined_prec, refined_rec, refined_fm, refined_iou = evaluate_block(instance['true_subgraph'],
                                                                            refined_pred_subgraph)

        logger.debug('-' * 5 + ' performance of refined prediction ' + '-' * 5)
        logger.debug('refined precision: {:.5f}'.format(refined_prec))
        logger.debug('refined recall   : {:.5f}'.format(refined_rec))
        logger.debug('refined f-measure: {:.5f}'.format(refined_fm))
        logger.debug('refined iou      : {:.5f}'.format(refined_iou))

        all_performance.append((prec, rec, fm, iou, refined_prec, refined_rec, refined_fm, refined_iou, run_time))

        # break # test 1 instance

    all_performance = np.array(all_performance)
    avg_performance = np.mean(all_performance, axis=0)
    logger.debug('-' * 5 + ' average performance ' + '-' * 5)
    logger.debug('average presision: {:.5f}'.format(avg_performance[0]))
    logger.debug('average recall   : {:.5f}'.format(avg_performance[1]))
    logger.debug('average f-measure: {:.5f}'.format(avg_performance[2]))
    logger.debug('average iou      : {:.5f}'.format(avg_performance[3]))
    logger.debug('avg refined prec : {:.5f}'.format(avg_performance[4]))
    logger.debug('avg refined rec  : {:.5f}'.format(avg_performance[5]))
    logger.debug('avg refined fm   : {:.5f}'.format(avg_performance[6]))
    logger.debug('avg refined iou  : {:.5f}'.format(avg_performance[7]))
    logger.debug('average run time : {:.5f}'.format(avg_performance[8]))


def train_mps(sparsity, trade_off, learning_rate, num_nodes, deg, normalize):
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/app2/ba'
    fn = 'train_{}_deg_{}.pkl'.format(num_nodes, deg)
    with open(os.path.join(path, fn), 'rb') as rfile:
        dataset = pickle.load(rfile)

    max_iter = 10000
    epsilon = 1e-3
    # write_to_dir = None
    write_to_dir = '/network/rit/lab/ceashpc/share_data/GraphOpt/log/ghtp/block_ba'

    paras = dataset, sparsity, trade_off, learning_rate, max_iter, epsilon, write_to_dir, num_nodes, deg, normalize
    run_dataset(paras)


if __name__ == '__main__':
    # sparsity = 50
    # trade_off = 0.0001
    # learning_rate = .5
    # num_nodes = 2000
    # deg = 3
    # normalize = False
    sparsity, trade_off, learning_rate, num_nodes, deg, normalize = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6].lower() == 'true'
    train_mps(sparsity, trade_off, learning_rate, num_nodes, deg, normalize)