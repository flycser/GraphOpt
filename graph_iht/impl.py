#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : alg_iht
# @Date     : 03/23/2019 21:25:07
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     : implementation of graph-structured iterative hard thresholding algorithm, Graph-IHT


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

from objs import EMS
from utils import evaluate, normalize_gradient, normalize
from data.utils import visual_grid_graph, visual_grid_graph_feature

from sparse_learning.proj_algo import head_proj
from sparse_learning.proj_algo import tail_proj


def optimize(instance, sparsity, learning_rate=0.01, max_iter=10, epsilon=1e-3):

    graph = instance['graph']

    true_subgraph = instance['subgraph']
    edges = np.array(graph.edges)
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    edge_weights = np.ones(num_edges)
    logger.debug('-' * 5 + ' related info ' + '-' * 5)
    logger.debug('sparsity: {:d}'.format(sparsity))
    logger.debug('max iteration: {:d}'.format(max_iter))
    logger.debug('number of nodes: {:d}'.format(num_nodes))
    logger.debug('number of edges: {:d}'.format(num_edges))
    logger.debug('number of nodes in true_subgraph: {:d}'.format(len(true_subgraph)))

    logger.debug('-' * 5 + ' start iterating ' + '-' * 5)
    start_time = time.time()
    acc_proj_time = 0.
    func = EMS(features=instance['features'], graph=graph)
    # current_x = func.get_init_x_random() # note, not stable
    current_x = func.get_init_x_zeros() + 0.00001
    for iter in range(max_iter):
        logger.debug('iteration: {:d}'.format(iter))
        iter_time = time.time()
        iter_proj_time = 0.

        grad_x = func.get_gradient(current_x)
        normalized_grad = normalize_gradient(current_x, grad_x) # fixme

        start_proj_time = time.time()
        re_head = head_proj(edges=edges, weights=edge_weights, x=normalized_grad, g=1, s=sparsity, budget=sparsity - 1., delta=1. / 169., max_iter=100, err_tol=1e-8, root=-1, pruning='strong', epsilon=1e-10, verbose=0)
        re_nodes, _, _ = re_head
        iter_proj_time += time.time() - start_proj_time

        omega_x = set(re_nodes)
        indicator_x = np.zeros(num_nodes)
        indicator_x[list(omega_x)] = 1.

        bx = current_x + learning_rate * normalized_grad * indicator_x

        start_proj_time = time.time()
        re_tail = tail_proj(edges=edges, weights=edge_weights, x=bx, g=1, s=sparsity, budget=sparsity - 1., nu=2.5, max_iter=100, err_tol=1e-8, root=-1, pruning='strong', verbose=0)
        re_nodes, _, _ = re_tail
        iter_proj_time += time.time() - start_proj_time
        acc_proj_time += time.time() - start_proj_time
        psi_x = set(re_nodes)

        prev_x = current_x
        current_x = np.zeros_like(current_x)
        current_x[list(psi_x)] = bx[list(psi_x)]
        current_x = normalize(current_x) # note, constrain current_x in [0, 1]

        logger.debug('function value: {:.5f}'.format(func.get_obj_val(current_x)))
        logger.debug('iteration time: {:.5f}'.format(time.time() - iter_time))
        logger.debug('iter projection time: {:.5f}'.format(iter_proj_time))
        logger.debug('acc projection time: {:.5f}'.format(acc_proj_time)) # accumulative projection time
        logger.debug('-' * 10)

        diff_norm_x = np.linalg.norm(current_x - prev_x)
        if diff_norm_x < epsilon:
            break

    run_time = time.time() - start_time
    logger.debug('final function value: {:.5f}'.format(func.get_obj_val(current_x)))
    logger.debug('run time of whole algorithm: {:.5f}'.format(run_time))
    logger.debug('accumulative projection time: {:.5f}'.format(acc_proj_time))

    return current_x

def run_instance(instance, sparsity, learning_rate=0.01, max_iter=200, epsilon=1e-3):
    """
    run graph-mp algrithm with fixed setting
    :param sparsity: sparsity upper bound of head and tail approximation
    :param rfn: input file
    :return:
    """

    opt_x = optimize(instance, sparsity, learning_rate, max_iter, epsilon)

    pred_subgraph = np.nonzero(opt_x)[0]
    prec, rec, fm, iou = evaluate(instance['subgraph'], pred_subgraph)
    logger.debug('precision: {:.5f}'.format(prec))
    logger.debug('recall: {:.5f}'.format(rec))
    logger.debug('f-measure: {:.5f}'.format(fm))
    logger.debug('iou: {:.5f}'.format(iou))

    visual_grid_graph_feature(10, 10, instance['features'])

    visual_grid_graph(10, 10, instance['subgraph'])

    visual_grid_graph(10, 10, pred_subgraph)


if __name__ == '__main__':
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/synthetic'
    fn = 'syn_mu_0.0_5.0_num_1_100_10.pkl'
    rfn = os.path.join(path, fn)
    # load dataset
    with open(os.path.join(rfn), 'rb') as rfile:
        dataset = pickle.load(rfile)

    instance = dataset[0]

    sparsity = 5
    run_instance(instance, sparsity)