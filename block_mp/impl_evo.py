#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : impl
# @Date     : 03/26/2019 17:16:16
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     : evolving patter detection


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
    true_subgraphs = instance['true_subgraphs']
    edges = np.array(graph.edges)
    start, end = instance['start'], instance['end']
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
    logger.debug('event interval: {:d}-{:d}'.format(start, end))

    logger.debug('-' * 5 + ' start iterating ' + '-' * 5)
    start_time = time.time()
    acc_proj_time = 0.
    func = GlobalEMS(features=instance['features'], trade_off=trade_off)
    current_x_array = func.get_init_x_random()
    for iter in range(max_iter):
        logger.debug('iteration: {:d}'.format(iter))
        iter_time = time.time()
        iter_proj_time = 0.

        omega_x_list = []
        for t in range(num_time_stamps):
            grad_x = func.get_gradient(current_x_array, t)
            current_x = current_x_array[t]
            normalized_grad = normalize_gradient(current_x, grad_x)

            start_proj_time = time.time()
            re_head = head_proj(edges=edges, weights=edge_weights, x=normalized_grad, g=1, s=sparsity, budget=sparsity - 1., delta=1. / 169., max_iter=100, err_tol=1e-8, root=-1, pruning='strong', epsilon=1e-10, verbose=0)
            re_nodes, _, _ = re_head
            iter_proj_time += time.time() - start_proj_time

            gamma_x = set(re_nodes)
            supp_x = set([ind for ind, _ in enumerate(current_x) if not _ == 0.])
            omega_x = gamma_x | supp_x
            omega_x_list.append(omega_x)

        bx_array = func.argmax_obj_with_proj(current_x_array, omega_x_list)

        start_proj_time = time.time()

        for t in range(num_time_stamps):
            bx = bx_array[t]
            re_tail = tail_proj(edges=edges, weights=edge_weights, x=bx, g=1, s=sparsity, budget=sparsity - 1., nu=2.5, max_iter=100, err_tol=1e-8, root=-1, pruning='strong', verbose=0)
            re_nodes, _, _ = re_tail
            iter_proj_time += time.time() - start_proj_time
            acc_proj_time += iter_proj_time
            psi_x = set(re_nodes)

            current_x = np.zeros_like(current_x_array[t])
            current_x[list(psi_x)] = bx[list(psi_x)]

            current_x_array[t] = current_x

        # post process

        logger.debug('function value: {:.5f}'.format(func.get_obj_val(current_x)))
        logger.debug('iteration time: {:.5f}'.format(time.time() - iter_time))
        logger.debug('iter projection time: {:.5f}'.format(iter_proj_time))
        logger.debug('acc projection time: {:.5f}'.format(acc_proj_time))  # accumulative projection time
        logger.debug('-' * 10)

        # diff_norm_x = np.linalg.norm(current_x - prev_x)
        # if diff_norm_x < epsilon:
        #     break

    run_time = time.time() - start_time
    logger.debug('final function value: {:.5f}'.format(func.get_obj_val(current_x)))
    logger.debug('run time of whole algorithm: {:.5f}'.format(run_time))
    logger.debug('accumulative projection time: {:.5f}'.format(acc_proj_time))

    return current_x


def run_instance():
    pass


if __name__ == '__main__':
    pass