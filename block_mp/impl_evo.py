#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : impl
# @Date     : 03/26/2019 17:16:16
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     : evolving patter detection


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

import numpy as np

from objs import GlobalEMS
from utils import evaluate, normalize_gradient
# from data.utils import visual_grid_graph, visual_grid_graph_feature

from sparse_learning.proj_algo import head_proj
from sparse_learning.proj_algo import tail_proj


def optimize(instance, sparsity, trade_off, max_iter=10, epsilon=1e-3):

    graph = instance['graph']
    true_subgraphs = instance['subgraphs']
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
    logger.debug('-' * 5 + ' related info ' + '-' * 5)
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
    true_obj_val, true_global_ems_val, true_penalty = func.get_obj_val(true_x_array)
    logger.debug('ground truth, obj value: {:.5f}, global ems value: {:.5f}, penalty: {:.5f}'.format(true_obj_val, true_global_ems_val, true_penalty))

    # current_x_array = func.get_init_x_random()
    current_x_array = func.get_init_x_zeros() + 1e-6
    for iter in range(max_iter):
        logger.debug('iteration: {:d}'.format(iter))
        prev_x_array = np.copy(current_x_array)
        iter_time = time.time()
        iter_proj_time = 0.

        omega_x_list = []
        for t in range(num_time_stamps):
            grad_x = func.get_gradient(current_x_array, t)
            current_x = current_x_array[t] if iter > 0 else np.zeros_like(current_x_array[t], dtype=np.float64)
            normalized_grad = normalize_gradient(current_x, grad_x)

            start_proj_time = time.time()
            re_head = head_proj(edges=edges, weights=edge_weights, x=normalized_grad, g=1, s=sparsity, budget=sparsity - 1., delta=1. / 169., max_iter=100, err_tol=1e-8, root=-1, pruning='strong', epsilon=1e-10, verbose=0)
            re_nodes, _, _ = re_head
            iter_proj_time += time.time() - start_proj_time

            gamma_x = set(re_nodes)
            supp_x = set([ind for ind, _ in enumerate(current_x) if not 0. == _])
            omega_x = gamma_x | supp_x
            omega_x_list.append(omega_x)

        bx_array = func.argmax_obj_with_proj(current_x_array, omega_x_list)

        for t in range(num_time_stamps):
            bx = bx_array[t]
            start_proj_time = time.time()
            re_tail = tail_proj(edges=edges, weights=edge_weights, x=bx, g=1, s=sparsity, budget=sparsity - 1., nu=2.5, max_iter=100, err_tol=1e-8, root=-1, pruning='strong', verbose=0)
            re_nodes, _, _ = re_tail
            iter_proj_time += time.time() - start_proj_time
            psi_x = set(re_nodes)

            current_x = np.zeros_like(current_x_array[t])
            current_x[list(psi_x)] = bx[list(psi_x)]

            current_x_array[t] = current_x

        acc_proj_time += iter_proj_time

        # post process

        obj_val, global_ems_val, penalty = func.get_obj_val(current_x_array)
        logger.debug('objective value: {:.5f}, global ems value: {:.5f}, penalty: {:.5f}'.format(obj_val, global_ems_val, penalty))
        logger.debug('iteration time: {:.5f}'.format(time.time() - iter_time))
        logger.debug('iter projection time: {:.5f}'.format(iter_proj_time))
        logger.debug('acc projection time: {:.5f}'.format(acc_proj_time))  # accumulative projection time
        logger.debug('-' * 10)

        diff_norm_x = np.linalg.norm(current_x_array - prev_x_array)
        if diff_norm_x < epsilon:
            break

    run_time = time.time() - start_time
    obj_val, global_ems_val, penalty = func.get_obj_val(current_x_array)
    logger.debug(
        'objective value: {:.5f}, global ems value: {:.5f}, penalty: {:.5f}'.format(obj_val, global_ems_val, penalty))
    logger.debug('run time of whole algorithm: {:.5f}'.format(run_time))
    logger.debug('accumulative projection time: {:.5f}'.format(acc_proj_time))

    return current_x_array


def run_instance(instance, sparsity, trade_off, max_iter=100, epsilon=1e-3):

    opt_x_array = optimize(instance, sparsity, trade_off, max_iter, epsilon)

    pred_subgraphs = []
    for opt_x in opt_x_array:
        pred_subgraph = np.nonzero(opt_x)[0]
        pred_subgraphs.append(pred_subgraph)

    # global_prec, global_rec, global_fm, global_iou, valid_global_iou, valid_avg_fm, avg_prec, avg_rec, avg_fm, avg_iou = evaluate(instance['subgraphs'], pred_subgraphs)
    global_prec, global_rec, global_fm, global_iou, valid_global_prec, valid_global_rec, valid_global_fm, valid_global_iou, _, _, _, _, _ = evaluate(instance['subgraphs'], pred_subgraphs)

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
    # logger.debug('average f-measure in signal interval: {:.5f}'.format(valid_avg_fm))
    # logger.debug('average precision: {:.5f}'.format(avg_prec))
    # logger.debug('average recall: {:.5f}'.format(avg_rec))
    # logger.debug('average f-measure: {:.5f}'.format(avg_fm))
    # logger.debug('average iou: {:.5f}'.format(avg_iou))


if __name__ == '__main__':
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
    trade_off = 0.01
    run_instance(instance, sparsity, trade_off)