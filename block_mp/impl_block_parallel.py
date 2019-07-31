#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : impl_block_parallel
# @Date     : 04/10/2019 14:00:12
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

import numpy as np
import networkx as nx

from objs import ParallelSumEMS
from utils import evaluate, normalize_gradient, relabel_nodes, relabel_edges, get_boundary_edge_x_dict

from sparse_learning.proj_algo import head_proj
from sparse_learning.proj_algo import tail_proj


def optimize(instance, sparsity, trade_off, max_iter, epsilon):

    graph = instance['graph']
    true_subgraph = instance['subgraph']
    features = instance['features']
    block_node_sets = instance['block_node_sets']
    block_boundary_edges_dict = instance['block_boundary_edges_dict']

    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    num_blocks = len(block_boundary_edges_dict)

    logger.debug('-' * 5 + ' related info ' + '-' * 5)
    logger.debug('algorithm: parallel graph block-structured matching pursuit')
    logger.debug('sparsity: {:d}'.format(sparsity))
    logger.debug('max iteration: {:d}'.format(max_iter))
    logger.debug('number of nodes: {:d}'.format(num_nodes))
    logger.debug('number of edges: {:d}'.format(num_edges))
    logger.debug('number of blocks: {:d}'.format(num_blocks))

    node_id_dict = relabel_nodes(block_node_sets) # label global node id with local node id
    relabeled_edge_sets = relabel_edges(graph, block_node_sets, node_id_dict) # relabel edges with block node id

    logger.debug('-' * 5 + ' start iterating ' + '-' * 5)
    start_time = time.time()
    acc_proj_time = 0.

    func = ParallelSumEMS(features=features, block_node_sets=block_node_sets, node_id_dict=node_id_dict, block_boundary_edges_dict=block_boundary_edges_dict, trade_off=trade_off)
    true_x = np.zeros(num_nodes)
    true_x[true_subgraph] = 1.
    true_x = np.array(true_x)
    true_obj_val, true_sum_ems_val, true_penalty = func.get_obj_val(true_x, block_boundary_edges_dict)
    logger.debug('ground truth, obj value: {:.5f}, sum ems value: {:.5f}, penalty: {:.5f}'.format(true_obj_val, true_sum_ems_val, true_penalty))

    current_x = func.get_init_x_zeros() + 1e-6 # the global vector
    for iter in range(max_iter):
        logger.debug('iteration: {:d}'.format(iter))
        prev_x = np.copy(current_x) # previous iteration vector
        iter_time = time.time()
        iter_proj_time = 0.

        omega_x_list = []
        for b in range(num_blocks):
            block_x = current_x[sorted(block_node_sets[b])] # get current block vector
            block_boundary_edge_x_dict = get_boundary_edge_x_dict(current_x, block_boundary_edges_dict[b], node_id_dict)  # (block node 1, global node 2) - value of node 2
            block_features = features[sorted(block_node_sets[b])]
            block_grad = func.get_gradient(block_x, block_features, block_boundary_edge_x_dict)

            block_x = block_x if iter > 0 else np.zeros_like(block_x, dtype=np.float64) # start from all 0s at iteration 0

            normalized_block_grad = normalize_gradient(block_x, block_grad)
            edges = np.array(relabeled_edge_sets[b])
            edge_weights = np.ones(len(edges))

            start_proj_time = time.time()
            re_head = head_proj(edges=edges, weights=edge_weights, x=normalized_block_grad, g=1, s=sparsity, budget=sparsity - 1., delta=1. / 169., max_iter=100, err_tol=1e-8, root=-1, pruning='strong', epsilon=1e-10, verbose=0)
            re_nodes, _, _ = re_head
            iter_proj_time += time.time() - start_proj_time
            block_gamma_x = set(re_nodes)
            block_supp_x = set([ind for ind, _ in enumerate(block_x) if not 0. == _])
            block_omega_x = block_gamma_x | block_supp_x
            omega_x_list.append(block_omega_x)

        bx = func.argmax_obj_with_proj_parallel(current_x, omega_x_list) # solve the sub-problem

        for b in range(num_blocks):
            edges = np.array(relabeled_edge_sets[b])
            edge_weights = np.ones(len(edges))
            sorted_node_set = sorted(block_node_sets[b])
            # block_bx = bx[sorted(block_node_sets[b])]
            block_bx = bx[sorted_node_set]
            start_proj_time = time.time()
            re_tail = tail_proj(edges=edges, weights=edge_weights, x=block_bx, g=1, s=sparsity, budget=sparsity - 1., nu=2.5, max_iter=100, err_tol=1e-8, root=-1, pruning='strong', verbose=0)
            re_nodes, _, _ = re_tail
            iter_proj_time += time.time() - start_proj_time
            psi_x = set(re_nodes)
            # block_x = np.zeros_like(current_x[block_node_sets[b]])
            block_x = np.zeros_like(current_x[sorted_node_set])
            block_x[list(psi_x)] = block_bx[list(psi_x)]
            # current_x[sorted(block_node_sets[b])] = block_x
            current_x[sorted_node_set] = block_x

        acc_proj_time += iter_proj_time

        # end of an iteration

        obj_val, sum_ems_val, penalty = func.get_obj_val(current_x, block_boundary_edges_dict)
        logger.debug('objective value: {:.5f}, sum ems value: {:.5f}, penalty: {:.5f}'.format(obj_val, sum_ems_val,penalty))
        logger.debug('iteration time: {:.5f}'.format(time.time() - iter_time))
        logger.debug('iter projection time: {:.5f}'.format(iter_proj_time))
        logger.debug('acc projection time: {:.5f}'.format(acc_proj_time))  # accumulative projection time
        logger.debug('-' * 10)

        diff_norm_x = np.linalg.norm(current_x - prev_x)
        if diff_norm_x < epsilon:
            print('difference {:.5f}'.format(diff_norm_x))
            break

    run_time = time.time() - start_time
    obj_val, sum_ems_val, penalty = func.get_obj_val(current_x, block_boundary_edges_dict)
    logger.debug('objective value: {:.5f}, sum ems value: {:.5f}, penalty: {:.5f}'.format(obj_val, sum_ems_val, penalty))
    logger.debug('run time of whole algorithm: {:.5f}'.format(run_time))
    logger.debug('accumulative projection time: {:.5f}'.format(acc_proj_time))

    return current_x


def post_process(graph, pred_sugraph):

    subgraph = nx.subgraph(graph, pred_sugraph)
    largest_connected_component = max(nx.connected_components(subgraph), key=len)

    refined_pred_subgraph = largest_connected_component # component datatype set

    return refined_pred_subgraph


def run_instance(instance, sparsity, trade_off, max_iter=10, epsilon=1e-3):
    opt_x = optimize(instance, sparsity, trade_off, max_iter, epsilon)

    pred_subgraph = np.nonzero(opt_x)[0]

    prec, rec, fm, iou = evaluate(instance['subgraph'], pred_subgraph)

    logger.debug('-' * 5 + ' raw performance ' + '-' * 5)
    logger.debug('precision: {:.5f}'.format(prec))
    logger.debug('recall   : {:.5f}'.format(rec))
    logger.debug('f-measure: {:.5f}'.format(fm))
    logger.debug('iou      : {:.5f}'.format(iou))

    refined_pred_subgraph = post_process(instance['graph'], pred_subgraph)
    prec, rec, fm, iou = evaluate(instance['subgraph'], refined_pred_subgraph)
    logger.debug('-' * 5 + ' refined performance ' + '-' * 5)
    logger.debug('precision: {:.5f}'.format(prec))
    logger.debug('recall   : {:.5f}'.format(rec))
    logger.debug('f-measure: {:.5f}'.format(fm))
    logger.debug('iou      : {:.5f}'.format(iou))

if __name__ == '__main__':
    # # path = '/network/rit/lab/ceashpc/share_data/GraphOpt/ijcai/app2/BA'
    # path = '/network/rit/lab/ceashpc/share_data/GraphOpt/ijcai/app2/CondMat'
    # fn = 'test_9.pkl'
    #
    # rfn = os.path.join(path, fn)
    # with open(rfn, 'rb') as rfile:
    #     dataset = pickle.load(rfile)
    #
    # instance = dataset[0]
    # # print(instance['graph'].number_of_nodes())
    # # print(len(instance['block_node_sets']))
    # sparsity = 534
    # trade_off = 0.05
    # run_instance(instance, sparsity, trade_off)

    path = '/network/rit/lab/ceashpc/share_data/BA/run2'
    fn = 'nodes_1000000_blocks_100_mu_3_subsize_50000_50000_deg_3_train_0.pkl'

    rfn = os.path.join(path, fn)
    with open(rfn, 'rb') as rfile:
        dataset = pickle.load(rfile)


    instance = dataset[0]
    new_instance = {}
    print(instance.keys())

    new_instance['features'] = instance['features']
    new_instance['subgraph'] = instance['true_subgraph']
    new_instance['graph'] = instance['graph']
    new_instance['block_boundary_edges_dict'] = instance['block_boundary_edges_dict']
    new_instance['block_node_sets'] = instance['nodes_set']

    sparsity = 3000
    trade_off = 0.001
    run_instance(new_instance, sparsity, trade_off)



