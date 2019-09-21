#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : dual_dblp_exp
# @Date     : 08/02/2019 11:22:01
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
# logger = logging.getLogger('fei')

import time
import pickle

import numpy as np
import networkx as nx

from impl_dual_2 import optimize
from utils import evaluate_block, post_process_block


DATASET = 'dblp'

def run_dataset(paras):

    dataset, sparsity, threshold, trade_off, learning_rate, max_iter, epsilon, write_to_dir = paras

    if not write_to_dir:
        logger = logging.getLogger('fei')
    else:
        log_fn = '{}_sparsity_{:d}_threshold_{}_trade_{}_lr_{}.txt'.format(DATASET, sparsity, threshold, trade_off, learning_rate)

        if os.path.isfile(os.path.join(write_to_dir, log_fn)):
            print('file exists !!!')
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

        true_subgraph = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

        # if normalize:
        #     logger.debug('feature normalized')
        #     instance['features'] = instance['features'] / np.max(instance['features'])

        opt_x, opt_y, run_time = optimize(instance, sparsity, threshold, trade_off, learning_rate, max_iter, epsilon, logger=logger)

        logger.debug('run time: {:.5f}'.format(run_time))

        raw_pred_subgraph_x = np.nonzero(opt_x)[0]

        print('rawx', raw_pred_subgraph_x)

        # prec, rec, fm, iou = evaluate_block(instance['true_subgraph'], raw_pred_subgraph_x)
        prec, rec, fm, iou = evaluate_block(true_subgraph, raw_pred_subgraph_x)

        logger.debug('-' * 5 + ' performance of x prediction ' + '-' * 5)
        logger.debug('precision: {:.5f}'.format(prec))
        logger.debug('recall   : {:.5f}'.format(rec))
        logger.debug('f-measure: {:.5f}'.format(fm))
        logger.debug('iou      : {:.5f}'.format(iou))

        raw_pred_subgraph_y = np.nonzero(opt_y)[0]

        print('rawy', raw_pred_subgraph_y)

        # prec, rec, fm, iou = evaluate_block(instance['true_subgraph'], raw_pred_subgraph_y)
        prec, rec, fm, iou = evaluate_block(true_subgraph, raw_pred_subgraph_y)

        logger.debug('-' * 5 + ' performance of y prediction ' + '-' * 5)
        logger.debug('precision: {:.5f}'.format(prec))
        logger.debug('recall   : {:.5f}'.format(rec))
        logger.debug('f-measure: {:.5f}'.format(fm))
        logger.debug('iou      : {:.5f}'.format(iou))

        # connected in first graph
        refined_pred_subgraph_x = post_process_block(instance['first_graph'], raw_pred_subgraph_x, dataset=DATASET)
        # refined_prec, refined_rec, refined_fm, refined_iou = evaluate_block(instance['true_subgraph'], refined_pred_subgraph_x)
        refined_prec, refined_rec, refined_fm, refined_iou = evaluate_block(true_subgraph, refined_pred_subgraph_x)

        print('refined x', refined_pred_subgraph_x)
        logger.debug('-' * 5 + ' performance of refined x prediction ' + '-' * 5)
        logger.debug('refined precision: {:.5f}'.format(refined_prec))
        logger.debug('refined recall   : {:.5f}'.format(refined_rec))
        logger.debug('refined f-measure: {:.5f}'.format(refined_fm))
        logger.debug('refined iou      : {:.5f}'.format(refined_iou))

        refined_pred_subgraph_y = post_process_block(instance['first_graph'], raw_pred_subgraph_y, dataset=DATASET)
        # refined_prec, refined_rec, refined_fm, refined_iou = evaluate_block(instance['true_subgraph'], refined_pred_subgraph_y)
        refined_prec, refined_rec, refined_fm, refined_iou = evaluate_block(true_subgraph, refined_pred_subgraph_y)

        print('refined y', refined_pred_subgraph_y)
        logger.debug('-' * 5 + ' performance of refined y prediction ' + '-' * 5)
        logger.debug('refined precision: {:.5f}'.format(refined_prec))
        logger.debug('refined recall   : {:.5f}'.format(refined_rec))
        logger.debug('refined f-measure: {:.5f}'.format(refined_fm))
        logger.debug('refined iou      : {:.5f}'.format(refined_iou))

        combine_pred_subgraph = raw_pred_subgraph_x & raw_pred_subgraph_y

        print('combine', combine_pred_subgraph)

        # combine_prec, combine_rec, combine_fm, combine_iou = evaluate_block(instance['true_subgraph'], raw_pred_subgraph_y)
        combine_prec, combine_rec, combine_fm, combine_iou = evaluate_block(true_subgraph, raw_pred_subgraph_y)

        logger.debug('-' * 5 + ' performance of combine prediction ' + '-' * 5)
        logger.debug('precision: {:.5f}'.format(combine_prec))
        logger.debug('recall   : {:.5f}'.format(combine_rec))
        logger.debug('f-measure: {:.5f}'.format(combine_fm))
        logger.debug('iou      : {:.5f}'.format(combine_iou))

        refined_combine_pred_subgraph = post_process_block(instance['first_graph'], combine_pred_subgraph, dataset=DATASET)

        print('refined combine', refined_combine_pred_subgraph)
        # refined_combine_prec, refined_combine_rec, refined_combine_fm, refined_combine_iou = evaluate_block(instance['true_subgraph'], raw_pred_subgraph_y)
        refined_combine_prec, refined_combine_rec, refined_combine_fm, refined_combine_iou = evaluate_block(true_subgraph, refined_combine_pred_subgraph)

        logger.debug('-' * 5 + ' performance of y prediction ' + '-' * 5)
        logger.debug('precision: {:.5f}'.format(refined_combine_prec))
        logger.debug('recall   : {:.5f}'.format(refined_combine_rec))
        logger.debug('f-measure: {:.5f}'.format(refined_combine_fm))
        logger.debug('iou      : {:.5f}'.format(refined_combine_iou))


        # all_performance.append((prec, rec, fm, iou, refined_prec, refined_rec, refined_fm, refined_iou, run_time))

        # break # test 1 instance

    # all_performance = np.array(all_performance)
    # avg_performance = np.mean(all_performance, axis=0)
    # logger.debug('-' * 5 + ' average performance ' + '-' * 5)
    # logger.debug('average presision: {:.5f}'.format(avg_performance[0]))
    # logger.debug('average recall   : {:.5f}'.format(avg_performance[1]))
    # logger.debug('average f-measure: {:.5f}'.format(avg_performance[2]))
    # logger.debug('average iou      : {:.5f}'.format(avg_performance[3]))
    # logger.debug('avg refined prec : {:.5f}'.format(avg_performance[4]))
    # logger.debug('avg refined rec  : {:.5f}'.format(avg_performance[5]))
    # logger.debug('avg refined fm   : {:.5f}'.format(avg_performance[6]))
    # logger.debug('avg refined iou  : {:.5f}'.format(avg_performance[7]))
    # logger.debug('average run time : {:.5f}'.format(avg_performance[8]))

def train(sparsity, threshold, trade_off, learning_rate):

    # path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/DBLP/DBLP_Citation_2014_May/DM'
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/DBLP/DBLP_Citation_2014_May/DB'
    # fn = 'dm_top30000_dataset.pkl'
    fn = 'db_top30000_dataset.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        instance = pickle.load(rfile)

    dataset = [instance]

    max_iter = 2
    epsilon = 1e-3
    write_to_dir = None

    paras = dataset, sparsity, threshold, trade_off, learning_rate, max_iter, epsilon, write_to_dir
    run_dataset(paras)




if __name__ == '__main__':

    sparsity = 300
    threshold = 0.004
    trade_off = 10.
    learning_rate = 1.

    train(sparsity, threshold, trade_off, learning_rate)