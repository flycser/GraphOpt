#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : evo_bwsn_exp
# @Date     : 05/01/2019 22:27:16
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

from impl_evo import optimize
from utils import evaluate_evo

from utils import post_process_evo


DATASET = 'bwsn'

def run_dataset(paras):
    dataset, sparsity, trade_off, learning_rate, max_iter, epsilon, write_to_dir, data_type, noise = paras

    if not write_to_dir:
        logger = logging.getLogger('fei')
    else:
        log_fn = '{}_sparsity_{:d}_trade_{}_lr_{}_{}_{}.txt'.format(DATASET, sparsity, trade_off, learning_rate, data_type, noise)
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

        # opt_x_array, run_time = optimize(instance, sparsity, trade_off, learning_rate, max_iter, epsilon, logger=None)
        opt_x_array, run_time = optimize(instance, sparsity, trade_off, learning_rate, max_iter, epsilon, logger=logger)

        logger.debug('run time: {:.5f}'.format(run_time))

        raw_pred_subgraphs = []
        for opt_x in opt_x_array:
            pred_subgraph = np.nonzero(opt_x)[0]
            raw_pred_subgraphs.append(pred_subgraph)

        global_prec, global_rec, global_fm, global_iou, valid_global_prec, valid_global_rec, valid_global_fm, valid_global_iou, _, _, _, _, _ = evaluate_evo(
            instance['subgraphs'], raw_pred_subgraphs)

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
        refined_global_prec, refined_global_rec, refined_global_fm, refined_global_iou, _, _, _, _, _, _, _, _, _ = evaluate_evo(instance['subgraphs'], refined_pred_subgraphs)

        logger.debug('-' * 5 + ' refined performance ' + '-' * 5)
        logger.debug('refined global precision: {:.5f}'.format(refined_global_prec))
        logger.debug('refined global recall   : {:.5f}'.format(refined_global_rec))
        logger.debug('refined global f-measure: {:.5f}'.format(refined_global_fm))
        logger.debug('refined global iou      : {:.5f}'.format(refined_global_iou))

        all_performance.append((global_prec, global_rec, global_fm, global_iou, refined_global_prec, refined_global_rec, refined_global_fm, refined_global_iou, run_time))

        # if i == 0:
        #     break

    all_performance = np.array(all_performance)
    avg_performance = np.mean(all_performance, axis=0)
    # print(all_performance.shape)
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


def train_mps(sparsity, trade_off, learning_rate, noise):

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/app1/bwsn'
    data_type = 'train'
    fn = '{}_noise_{}.pkl'.format(data_type, noise)
    rfn = os.path.join(path, fn)
    with open(rfn, 'rb') as rfile:
        dataset = pickle.load(rfile)

    # sparsity_list = [800, 850, 900, 950, 1000]
    # sparsity_list = [1000]
    # trade_off_list = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
    # trade_off_list = [0.1]
    # learning_rate_list = [1., 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    # learning_rate_list = [0.01]
    max_iter = 2000
    epsilon = 1e-3
    write_to_dir = '/network/rit/lab/ceashpc/share_data/GraphOpt/log/iht/evo_bwsn'

    paras = dataset, sparsity, trade_off, learning_rate, max_iter, epsilon, write_to_dir, data_type, noise
    run_dataset(paras)
    # input_paras = []
    # for sparsity in sparsity_list:
    #     for trade_off in trade_off_list:
    #         for learning_rate in learning_rate_list:

                # paras = dataset, sparsity, trade_off, learning_rate, max_iter, epsilon, write_to_dir
                # input_paras.append(paras)

    # num_prcs = 20
    # pool = multiprocessing.Pool(processes=num_prcs)
    # pool.map(run_dataset, input_paras)
    # pool.close()
    # pool.join()


def test(sparsity, trade_off, learning_rate, noise):

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/app1/bwsn'
    data_type = 'test'
    fn = '{}_noise_{}.pkl'.format(data_type, noise)
    rfn = os.path.join(path, fn)
    with open(rfn, 'rb') as rfile:
        dataset = pickle.load(rfile)

    new_dataset = []

    for instance in dataset:
        if 'subgraphs' in instance.keys():
            instance['true_subgraphs'] = instance.pop('subgraphs')
            # print(instance.keys())
            new_dataset.append(instance)

    max_iter = 2000
    epsilon = 1e-3
    # write_to_dir = '/network/rit/lab/ceashpc/share_data/GraphOpt/log/iht/evo_bwsn'
    write_to_dir = None


    print(sparsity, trade_off, learning_rate, max_iter, noise, fn)

    paras = new_dataset, sparsity, trade_off, learning_rate, max_iter, epsilon, write_to_dir, data_type, noise

    run_dataset(paras)


if __name__ == '__main__':
    # sparsity, trade_off, learning_rate, noise = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])

    # train_mps(sparsity, trade_off, learning_rate, noise)

    sparsity, trade_off, learning_rate, noise = 1000, 0.001, 1., 10

    test(sparsity, trade_off, learning_rate, noise)