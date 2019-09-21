#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : impl
# @Date     : 03/26/2019 17:16:41
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

from objs import GlobalEMS
from utils import evaluate_evo, normalize, normalize_gradient, post_process_evo

from sparse_learning.proj_algo import head_proj
from sparse_learning.proj_algo import tail_proj


def optimize(instance, sparsity, trade_off, learning_rate, max_iter, epsilon=1e-3, logger=None):

    graph = instance['graph']
    true_subgraphs = instance['true_subgraphs']
    edges = np.array(graph.edges)
    tag = False
    start = end = 0
    for t, subgraph in enumerate(true_subgraphs):
        if subgraph and not tag:
            start = t
            tag = True
        if not subgraph and tag:
            end = t - 1
            tag = False

    num_time_stamps = len(true_subgraphs)
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    edge_weights = np.ones(num_edges)

    if logger:
        logger.debug('-' * 5 + ' related info ' + '-' * 5)
        logger.debug('algorithm: graph block-structured IHT')
        logger.debug('sparsity: {:d}'.format(sparsity))
        logger.debug('max iteration: {:d}'.format(max_iter))
        logger.debug('number of nodes: {:d}'.format(num_nodes))
        logger.debug('number of edges: {:d}'.format(num_edges))
        logger.debug('number of time stamps: {:d}'.format(num_time_stamps))
        logger.debug('signal interval: [{:d}, {:d}]'.format(start, end))
        logger.debug('-' * 5 + ' start iterating ' + '-' * 5)

    start_time = time.time()
    acc_proj_time = 0.
    func = GlobalEMS(features=instance['features'], trade_off=trade_off)
    true_x_array = []
    for true_subgraph in true_subgraphs:
        true_x = np.zeros(num_nodes)
        true_x[true_subgraph] = 1.
        true_x_array.append(true_x)
    true_x_array = np.array(true_x_array)
    true_obj_val, true_gloabl_ems_val, true_penalty = func.get_obj_val(true_x_array)
    if logger:
        logger.debug('ground truth, obj value: {:.5f}, global ems value: {:.5f}, penalty: {:.5f}'.format(true_obj_val, true_gloabl_ems_val, true_penalty))

    current_x_array = func.get_init_x_zeros() + 1e-6
    # current_x_array = true_x_array
    # print('start from grount truth')
    for iter in range(max_iter):
        if logger:
            logger.debug('iteration: {:d}'.format(iter))
        prev_x_array = np.copy(current_x_array)
        iter_time = time.time()
        iter_proj_time = 0.

        omega_x_list = []
        for t in range(num_time_stamps):
            grad_x = func.get_gradient(current_x_array, t)
            print(grad_x)
            current_x = current_x_array[t] if iter > 0 else np.zeros_like(current_x_array[t], dtype=np.float64)
            normalized_grad = normalize_gradient(current_x, grad_x)
            start_proj_time = time.time()
            re_head = head_proj(edges=edges, weights=edge_weights, x=normalized_grad, g=1, s=sparsity, budget=sparsity - 1., delta=1. / 169., max_iter=100, err_tol=1e-8, root=-1, pruning='strong', epsilon=1e-10, verbose=0)
            re_nodes, _, _ = re_head
            iter_proj_time += time.time() - start_proj_time
            omega_x = set(re_nodes)
            omega_x_list.append(omega_x)
            print(sorted(omega_x))

        # update style 1
        # update all blocks simultaneously at each iteration
        # this style is analogous to gradient descent
        # pls refer to Andrew Ng' Machine Learning course
        bx_array = np.zeros_like(current_x_array)  # update
        for t in range(num_time_stamps):
            indicator_x = np.zeros(num_nodes)
            indicator_x[list(omega_x_list[t])] = 1.
            bx_array[t] = current_x_array[t] + learning_rate * func.get_gradient(current_x_array, t) * indicator_x # since bx as an intermediate variable, we can update gradient simultaneously

        for t in range(num_time_stamps):
            bx = bx_array[t]
            start_proj_time = time.time()
            re_tail = tail_proj(edges=edges, weights=edge_weights, x=bx, g=1, s=sparsity, budget=sparsity - 1., nu=2.5, max_iter=100, err_tol=1e-8, root=-1, pruning='strong', verbose=0)
            re_nodes, _, _ = re_tail
            iter_proj_time += time.time() - start_proj_time
            psi_x = set(re_nodes)

            current_x = np.zeros_like(current_x_array[t])
            current_x[list(psi_x)] = bx[list(psi_x)]
            current_x = normalize(current_x) # note, restrict current_x in [0, 1]

            current_x_array[t] = current_x

            # print(t, sorted(np.nonzero(current_x)))

        acc_proj_time += iter_proj_time

        if logger:
            obj_val, global_ems_val, penalty = func.get_obj_val(current_x_array)
            logger.debug('objective value: {:.5f}, global ems value: {:.5f}, penalty: {:.5f}'.format(obj_val, global_ems_val, penalty))
            logger.debug('iteration time: {:.5f}'.format(time.time() - iter_time))
            logger.debug('iter projection time: {:.5f}'.format(iter_proj_time))
            logger.debug('acc projection time: {:.5f}'.format(acc_proj_time))  # accumulative projection time
            logger.debug('-' * 10)

        diff_norm_x = np.linalg.norm(current_x_array - prev_x_array)
        if logger:
            logger.debug('difference norm x: {:.5f}'.format(diff_norm_x))
        if diff_norm_x < epsilon:
            break

    run_time = time.time() - start_time
    if logger:
        obj_val, global_ems_val, penalty = func.get_obj_val(current_x_array)
        logger.debug('objective value: {:.5f}, global ems value: {:.5f}, penalty: {:.5f}'.format(obj_val, global_ems_val, penalty))
        logger.debug('run time of whole algorithm: {:.5f}'.format(run_time))
        logger.debug('accumulative projection time: {:.5f}'.format(acc_proj_time))

    return current_x_array, run_time


def run_instance(instance, sparsity, trade_off, learning_rate, max_iter=2000, epsilon=1e-3, logger=None):

    opt_x_array, _ = optimize(instance, sparsity, trade_off, learning_rate, max_iter, epsilon)

    raw_pred_subgraphs = []
    for opt_x in opt_x_array:
        pred_subgraph = np.nonzero(opt_x)[0]
        raw_pred_subgraphs.append(pred_subgraph)

    global_prec, global_rec, global_fm, global_iou, valid_global_prec, valid_global_rec, valid_global_fm, valid_global_iou, _, _, _, _, _ = evaluate_evo(
        instance['true_subgraphs'], raw_pred_subgraphs)

    logger.debug('-' * 5 + 'performance in the whole interval' + '-' * 5)
    logger.debug('global precision: {:.5f}'.format(global_prec))
    logger.debug('global recall   : {:.5f}'.format(global_rec))
    logger.debug('global f-measure: {:.5f}'.format(global_fm))
    logger.debug('global iou      : {:.5f}'.format(global_iou))
    logger.debug('-' * 5 + 'performance in the interval with signals' + '-' * 5)
    logger.debug('global precision: {:.5f}'.format(valid_global_prec))
    logger.debug('global recall   : {:.5f}'.format(valid_global_rec))
    logger.debug('global f-measure: {:.5f}'.format(valid_global_fm))
    logger.debug('global iou      : {:.5f}'.format(valid_global_iou))

    refined_pred_subgraphs = post_process_evo(instance['graph'], raw_pred_subgraphs)
    global_prec, global_rec, global_fm, global_iou, valid_global_prec, valid_global_rec, valid_global_fm, valid_global_iou, _, _, _, _, _ = evaluate_evo(
        instance['true_subgraphs'], refined_pred_subgraphs)

    logger.debug('-' * 5 + ' refined performance ' + '-' * 5)
    logger.debug('refined global precision: {:.5f}'.format(global_prec))
    logger.debug('refined global recall   : {:.5f}'.format(global_rec))
    logger.debug('refined global f-measure: {:.5f}'.format(global_fm))
    logger.debug('refined global iou      : {:.5f}'.format(global_iou))


if __name__ == '__main__':
    # num_instances = 1
    # num_nodes = 100
    # num_time_stamps = 9
    # num_time_stamps_signal = 5
    # start_time_stamps = num_time_stamps / 2 - num_time_stamps_signal / 2
    # end_time_stamps = num_time_stamps / 2 + num_time_stamps_signal / 2
    # num_nodes_subgraph_min = 10
    # num_nodes_subgraph_max = 20
    # overlap_ratio = 0.8
    # mu_0 = 0.
    # mu_1 = 5.
    #
    # path = '/network/rit/lab/ceashpc/share_data/GraphOpt/synthetic'
    # fn = 'syn_mu_{:.1f}_{:.1f}_num_{:d}_{:d}_{:d}_{:d}_time_{:d}_{:d}_{:d}.pkl'.format(mu_0, mu_1, num_instances, num_nodes, num_nodes_subgraph_min, num_nodes_subgraph_max, start_time_stamps, end_time_stamps, num_time_stamps)
    # rfn = os.path.join(path, fn)
    # # load dataset
    # with open(os.path.join(rfn), 'rb') as rfile:
    #     dataset = pickle.load(rfile)
    #
    # instance = dataset[0]
    #
    # sparsity = 10
    # trade_off = 0.01
    # learning_rate = .05
    # run_instance(instance, sparsity, trade_off, learning_rate)

    num_instances = 1
    num_nodes = 100
    num_time_stamps = 9
    num_time_stamps_signal = 5
    start_time_stamps = num_time_stamps / 2 - num_time_stamps_signal / 2
    end_time_stamps = num_time_stamps / 2 + num_time_stamps_signal / 2
    num_nodes_subgraph_min = 10
    num_nodes_subgraph_max = 20
    overlap_ratio = 0.8
    mu_0 = 0.
    mu_1 = 5.

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/synthetic'
    fn = 'syn_mu_{:.1f}_{:.1f}_num_{:d}_{:d}_{:d}_{:d}_time_{:d}_{:d}_{:d}.pkl'.format(mu_0, mu_1, num_instances, num_nodes, num_nodes_subgraph_min, num_nodes_subgraph_max, start_time_stamps, end_time_stamps, num_time_stamps)
    rfn = os.path.join(path, fn)
    # load dataset
    with open(os.path.join(rfn), 'rb') as rfile:
        dataset = pickle.load(rfile)

    instance = dataset[0]

    sparsity = 10
    trade_off = 0.005
    learning_rate = 1.
    max_iter = 2000
    epsilon = 1e-3
    run_instance(instance, sparsity, trade_off, learning_rate)
