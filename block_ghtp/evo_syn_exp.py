#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : evo_syn_exp
# @Date     : 04/26/2019 18:25:44
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     : experiment on synthetic dataset for evolution application


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

from impl_evo import optimize
from utils import evaluate_evo

from utils import post_process_evo


DATASET = 'syn'

def run_dataset(paras):
    dataset, sparsity, trade_off, learning_rate, max_iter, epsilon, write_to_dir, mu = paras

    if not write_to_dir:
        logger = logging.getLogger('fei')
    else:
        log_fn = '{}_mu_{}_sparsity_{:d}_trade_{}_lr_{}_curve.txt'.format(DATASET, mu, sparsity, trade_off, learning_rate)

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

        opt_x_array, run_time = optimize(instance, sparsity, trade_off, learning_rate, max_iter, epsilon, logger)

        raw_pred_subgraphs = []
        for opt_x in opt_x_array:
            pred_subgraph = np.nonzero(opt_x)[0]
            raw_pred_subgraphs.append(pred_subgraph)

        global_prec, global_rec, global_fm, global_iou, valid_global_prec, valid_global_rec, valid_global_fm, valid_global_iou, _, _, _, _, _ = evaluate_evo(instance['true_subgraphs'], raw_pred_subgraphs)

        logger.debug('-' * 5 + ' performance in the whole interval ' + '-' * 5)
        logger.debug('global precision: {:.5f}'.format(global_prec))
        logger.debug('global recall   : {:.5f}'.format(global_rec))
        logger.debug('global f-measure: {:.5f}'.format(global_fm))
        logger.debug('global iou      : {:.5f}'.format(global_iou))
        logger.debug('-' * 5 + ' performance in the interval with signals ' + '-' * 5)
        logger.debug('global precision: {:.5f}'.format(valid_global_prec))
        logger.debug('global recall   : {:.5f}'.format(valid_global_rec))
        logger.debug('global f-measure: {:.5f}'.format(valid_global_fm))
        logger.debug('global iou      : {:.5f}'.format(valid_global_iou))

        refined_pred_subgraphs = post_process_evo(instance['graph'], raw_pred_subgraphs, dataset=DATASET)
        refined_global_prec, refined_global_rec, refined_global_fm, refined_global_iou, _, _, _, _, _, _, _, _, _ = evaluate_evo(instance['true_subgraphs'], refined_pred_subgraphs)

        logger.debug('-' * 5 + ' refined performance ' + '-' * 5)
        logger.debug('refined global precision: {:.5f}'.format(refined_global_prec))
        logger.debug('refined global recall   : {:.5f}'.format(refined_global_rec))
        logger.debug('refined global f-measure: {:.5f}'.format(refined_global_fm))
        logger.debug('refined global iou      : {:.5f}'.format(refined_global_iou))

        all_performance.append((global_prec, global_rec, global_fm, global_iou, refined_global_prec, refined_global_rec, refined_global_fm, refined_global_iou, run_time))

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


def train_mps():
    mu = 5
    data_type = 'train'
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/app1/synthetic'
    # fn = 'nodes_3000_windows_7_mu_{}_subsize_100_300_range_1_5_overlap_0.5_m_{}.pkl'.format(mu, data_type)
    fn = '{}_mu_{}.pkl'.format(data_type, mu)

    rfn = os.path.join(path, fn)
    with open(rfn, 'rb') as rfile:
        dataset = pickle.load(rfile)

    # sparsity_list = [100, 125, 150, 175, 200]
    sparsity_list = [150]
    # trade_off_list = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    trade_off_list = [0.001]
    # learning_rate_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.]
    learning_rate_list = [1.]
    max_iter = 2000
    epsilon = 1e-3
    # write_to_dir = '/network/rit/lab/ceashpc/share_data/GraphOpt/log/ghtp/evo_syn'
    write_to_dir = None

    input_paras = []
    for sparsity in sparsity_list:
        for trade_off in trade_off_list:
            for learning_rate in learning_rate_list:
                paras = dataset, sparsity, trade_off, learning_rate, max_iter, epsilon, write_to_dir, mu
                run_dataset(paras)
                # input_paras.append(paras)

    # num_prcs = 40
    # pool = multiprocessing.Pool(processes=num_prcs)
    # pool.map(run_dataset, input_paras)
    # pool.close()
    # pool.join()


def test():
    mu = 3
    data_type = 'train'
    # path = '/network/rit/lab/ceashpc/share_data/GraphOpt/app1/synthetic'
    # fn = 'nodes_3000_windows_7_mu_{}_subsize_100_300_range_1_5_overlap_0.5_m_{}.pkl'.format(mu, data_type)

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/app1/synthetic'
    fn = '{}_mu_{}.pkl'.format(data_type, mu)

    rfn = os.path.join(path, fn)
    with open(rfn, 'rb') as rfile:
        dataset = pickle.load(rfile)

    sparsity = 125
    trade_off = 0.0001
    learning_rate = 1.
    max_iter = 2000
    epsilon = 1e-3
    write_to_dir = None

    print(len(dataset))
    # print(dataset[0]['graph'].number_of_nodes())
    # print(dataset[0]['graph'].number_of_edges())

    paras = dataset, sparsity, trade_off, learning_rate, max_iter, epsilon, write_to_dir, mu

    run_dataset(paras)


if __name__ == '__main__':

    # train_mps()

    test()