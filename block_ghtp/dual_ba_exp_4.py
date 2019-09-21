#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : dual_ba_exp_4
# @Date     : 09/05/2019 22:25:53
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     : ems implement density projection
# @Desc     : xtw+xtAx/xt1 will include some nodes whose weights are postive and degrees are larger than nodes in true subgraph, this can not be avoided


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

from impl_dual_5 import optimize
from utils import evaluate_block, post_process_block

DATASET = 'ba'

def run_dataset(paras):
    dataset, sparsity, threshold, trade_off, learning_rate, max_iter, epsilon, write_to_dir = paras

    # print log setting
    if not write_to_dir:
        logger = logging.getLogger('fei')
    else:
        log_fn = '{}_sparsity_{:d}_threshold_{}_trade_off_{}_lr_{}.txt'.format(DATASET, sparsity, threshold, trade_off, learning_rate)

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

        opt_x, opt_y, run_time = optimize(instance, sparsity, threshold, trade_off, learning_rate, max_iter, epsilon, logger=logger)

        logger.debug('run time: {:.5f}'.format(run_time))

        second_graph = instance['second_graph']

        raw_pred_subgraph_x = np.nonzero(opt_x)[0]

        x_prec, x_rec, x_fm, x_iou = evaluate_block(instance['true_subgraph'], raw_pred_subgraph_x)

        logger.debug('-' * 5 + ' performance of x prediction ' + '-' * 5)
        logger.debug('precision: {:.5f}'.format(x_prec))
        logger.debug('recall   : {:.5f}'.format(x_rec))
        logger.debug('f-measure: {:.5f}'.format(x_fm))
        logger.debug('iou      : {:.5f}'.format(x_iou))
        logger.debug('raw density second graph: {:.5f}'.format(nx.density(nx.subgraph(second_graph, list(raw_pred_subgraph_x)))))

        raw_pred_subgraph_y = np.nonzero(opt_y)[0]

        print('length y', len(raw_pred_subgraph_y))

        y_prec, y_rec, y_fm, y_iou = evaluate_block(instance['true_subgraph'], raw_pred_subgraph_y)

        logger.debug('-' * 5 + ' performance of y prediction ' + '-' * 5)
        logger.debug('precision: {:.5f}'.format(y_prec))
        logger.debug('recall   : {:.5f}'.format(y_rec))
        logger.debug('f-measure: {:.5f}'.format(y_fm))
        logger.debug('iou      : {:.5f}'.format(y_iou))
        logger.debug('raw density second graph: {:.5f}'.format(nx.density(nx.subgraph(second_graph, list(raw_pred_subgraph_y)))))

        # connected in first graph
        connected_pred_subgraph_x = post_process_block(instance['first_graph'], raw_pred_subgraph_x, dataset=DATASET)
        connected_x_prec, connected_x_rec, connected_x_fm, connected_x_iou = evaluate_block(instance['true_subgraph'],connected_pred_subgraph_x)

        logger.debug('-' * 5 + ' performance of connected x prediction ' + '-' * 5)
        logger.debug('refined precision: {:.5f}'.format(connected_x_prec))
        logger.debug('refined recall   : {:.5f}'.format(connected_x_rec))
        logger.debug('refined f-measure: {:.5f}'.format(connected_x_fm))
        logger.debug('refined iou      : {:.5f}'.format(connected_x_iou))
        logger.debug('connected x density second graph: {:.5f}'.format(nx.density(nx.subgraph(second_graph, connected_pred_subgraph_x))))

        connected_pred_subgraph_y = post_process_block(instance['first_graph'], raw_pred_subgraph_y, dataset=DATASET)
        connected_y_prec, connected_y_rec, connected_y_fm, connected_y_iou = evaluate_block(instance['true_subgraph'],connected_pred_subgraph_y)

        logger.debug('-' * 5 + ' performance of connected y prediction ' + '-' * 5)
        logger.debug('refined precision: {:.5f}'.format(connected_y_prec))
        logger.debug('refined recall   : {:.5f}'.format(connected_y_rec))
        logger.debug('refined f-measure: {:.5f}'.format(connected_y_fm))
        logger.debug('refined iou      : {:.5f}'.format(connected_y_iou))
        logger.debug('refined y density second graph: {:.5f}'.format(nx.density(nx.subgraph(second_graph, connected_pred_subgraph_y))))

        # print(type(raw_pred_subgraph_x))
        # print(type(raw_pred_subgraph_y))

        intersect_pred_subgraph = set(list(raw_pred_subgraph_x)) & set(list(raw_pred_subgraph_y))

        intersect_prec, intersect_rec, intersect_fm, intersect_iou = evaluate_block(instance['true_subgraph'], intersect_pred_subgraph)

        logger.debug('-' * 5 + ' intersect performance of prediction ' + '-' * 5)
        logger.debug('precision: {:.5f}'.format(intersect_prec))
        logger.debug('recall   : {:.5f}'.format(intersect_rec))
        logger.debug('f-measure: {:.5f}'.format(intersect_fm))
        logger.debug('iou      : {:.5f}'.format(intersect_iou))
        logger.debug('intersect density second graph: {:.5f}'.format(nx.density(nx.subgraph(second_graph, intersect_pred_subgraph))))

        connected_intersect_pred_subgraph = post_process_block(instance['first_graph'], intersect_pred_subgraph, dataset=DATASET)
        connected_intersect_prec, connected_intersect_rec, connected_intersect_fm, connected_intersect_iou = evaluate_block(instance['true_subgraph'], connected_intersect_pred_subgraph)

        logger.debug('-' * 5 + ' connected intersect performance of prediction ' + '-' * 5)
        logger.debug('precision: {:.5f}'.format(connected_intersect_prec))
        logger.debug('recall   : {:.5f}'.format(connected_intersect_rec))
        logger.debug('f-measure: {:.5f}'.format(connected_intersect_fm))
        logger.debug('iou      : {:.5f}'.format(connected_intersect_iou))
        logger.debug('connected intersect density second graph: {:.5f}'.format(nx.density(nx.subgraph(second_graph, connected_intersect_pred_subgraph))))

        union_pred_subgraph = set(list(raw_pred_subgraph_x)) | set(list(raw_pred_subgraph_y))
        union_prec, union_rec, union_fm, union_iou = evaluate_block(instance['true_subgraph'], union_pred_subgraph)

        logger.debug('-' * 5 + ' union performance of prediction ' + '-' * 5)
        logger.debug('precision: {:.5f}'.format(union_prec))
        logger.debug('recall   : {:.5f}'.format(union_rec))
        logger.debug('f-measure: {:.5f}'.format(union_fm))
        logger.debug('iou      : {:.5f}'.format(union_iou))
        logger.debug('union density second graph: {:.5f}'.format(nx.density(nx.subgraph(second_graph, union_pred_subgraph))))

        connected_union_pred_subgraph = post_process_block(instance['first_graph'], union_pred_subgraph, dataset=DATASET)
        connected_union_prec, connected_union_rec, connected_union_fm, connected_union_iou = evaluate_block(instance['true_subgraph'], connected_union_pred_subgraph)

        logger.debug('-' * 5 + ' connected union performance of prediction ' + '-' * 5)
        logger.debug('precision: {:.5f}'.format(connected_union_prec))
        logger.debug('recall   : {:.5f}'.format(connected_union_rec))
        logger.debug('f-measure: {:.5f}'.format(connected_union_fm))
        logger.debug('iou      : {:.5f}'.format(connected_union_iou))
        logger.debug('connected union density second graph: {:.5f}'.format(nx.density(nx.subgraph(second_graph, connected_union_pred_subgraph))))



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



def train(sparsity, threshold, trade_off, learning_rate, case_id):

    num_nodes = 1000
    first_deg = 3
    second_deg = 10
    subgraph_size = 100
    mu = 3
    restart = 0.3


    # restart smaller, shape will not round, not be biased for eventallpairs+ algorithm
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic'
    # path = '/Users/fei/Downloads/Tmp/dcs_syn_dataset/data'
    # fn = 'dual_nodes_{}_deg1_{}_deg2_{}_subsize_{}_{}.pkl'.format(num_nodes, first_deg, second_deg, subgraph_size, case_id)
    fn = 'dual_mu_{}_nodes_{}_deg1_{}_deg2_{}_subsize_{}_restart_{}_{}.pkl'.format(mu, num_nodes, first_deg, second_deg, subgraph_size, restart, case_id)

    with open(os.path.join(path, fn), 'rb') as rfile:
        instance = pickle.load(rfile)

    print('case : {}'.format(case_id))

    dataset = [instance]

    max_iter = 1
    epsilon = 1e-3
    write_to_dir = None

    paras = dataset, sparsity, threshold, trade_off, learning_rate, max_iter, epsilon, write_to_dir
    run_dataset(paras)


if __name__ == '__main__':

    sparsity = 50
    threshold = 0.05
    trade_off = 1.
    learning_rate = 1.
    case_id = 0
    # case_id = int(sys.argv[1])

    train(sparsity, threshold, trade_off, learning_rate, case_id)