#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : block_wikivote_serial_exp
# @Date     : 07/17/2019 21:25:44
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

from impl_block_serial import optimize
from utils import evaluate_block
from utils import post_process_block

DATASET = 'wikivote'

def run_dataset(paras):
    dataset, sparsity, trade_off, learning_rate, max_iter, epsilon, write_to_dir = paras

    if not write_to_dir:
        logger = logging.getLogger('fei')
    else:
        log_fn = '{}_sparsity_{:d}_trade_{}_lr_{}_serial.txt'.format(DATASET, sparsity, trade_off, learning_rate)

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

        opt_x, run_time = optimize(instance, sparsity, trade_off, learning_rate, max_iter, epsilon, logger)

        logger.debug('run time: {:.5f}'.format(run_time))

        raw_pred_subgraph = np.nonzero(opt_x)[0]

        prec, rec, fm, iou = evaluate_block(instance['true_subgraph'], raw_pred_subgraph)

        logger.debug('-' * 5 + ' performance of raw prediction ' + '-' * 5)
        logger.debug('precision: {:.5f}'.format(prec))
        logger.debug('recall   : {:.5f}'.format(rec))
        logger.debug('f-measure: {:.5f}'.format(fm))
        logger.debug('iou      : {:.5f}'.format(iou))

        refined_pred_subgraph = post_process_block(instance['graph'], raw_pred_subgraph, dataset=DATASET)
        refined_prec, refined_rec, refined_fm, refined_iou = evaluate_block(instance['true_subgraph'], refined_pred_subgraph)

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



def train_mps(sparsity, trade_off, learning_rate):
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/app2/wikivote'
    fn = 'train.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        dataset = pickle.load(rfile)

    # sparsity_list = [160]
    # trade_off_list = [0.001]
    # learning_rate_list = [1.]
    max_iter = 10
    epsilon = 1e-3
    # write_to_dir = None
    write_to_dir = '/network/rit/lab/ceashpc/share_data/GraphOpt/log/ghtp/block_wikivote_3'

    paras = dataset, sparsity, trade_off, learning_rate, max_iter, epsilon, write_to_dir
    run_dataset(paras)
    # input_paras = []
    # for sparsity in sparsity_list:
    #     for trade_off in trade_off_list:
    #         for learning_rate in learning_rate_list:
    #             paras = dataset, sparsity, trade_off, learning_rate, max_iter, epsilon, write_to_dir
    #             input_paras.append(paras)
    #
    # num_prcs = 1
    # pool = multiprocessing.Pool(processes=num_prcs)
    # pool.map(run_dataset, input_paras)
    # pool.close()
    # pool.join()



def test():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/app2/wikivote'
    fn = 'train.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        dataset = pickle.load(rfile)

    sparsity = 500
    trade_off = 0.001
    learning_rate = 1.
    max_iter = 10
    epsilon = 1e-3
    write_to_dir = None

    paras = dataset, sparsity, trade_off, learning_rate, max_iter, epsilon, write_to_dir

    run_dataset(paras)


if __name__ == '__main__':
    sparsity, trade_off, learning_rate = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    train_mps(sparsity, trade_off, learning_rate)