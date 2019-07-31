#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : impl_block_3
# @Date     : 07/22/2019 14:50:49
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


import time

import numpy as np

from objs import BlockSumEMS
from utils import relabel_nodes, relabel_edges, get_boundary_xs, normalize, normalize_gradient

from sparse_learning.proj_algo import head_proj
from sparse_learning.proj_algo import tail_proj


def optimize(instance, sparsity, trade_off, learning_rate, max_iter, epsilon=1e-3, logger=None):

    graph = instance['graph']
    true_subgraph = instance['true_subgraph']
    features = instance['features']

    num_blocks = len(instance['nodes_set'])
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    nodes_set = instance['nodes_set']
    boundary_edges_dict = instance['block_boundary_edges_dict']

    nodes_id_dict = relabel_nodes(nodes_set)
    relabeled_edges_set = relabel_edges(graph, nodes_set, nodes_id_dict)

    if logger:
        logger.debug('-' * 5 + ' related info ' + '-' * 5)
        logger.debug('algorithm: graph block-structured IHT')
        logger.debug('sparsity: {:d}'.format(sparsity))
        logger.debug('max iteration: {:d}'.format(max_iter))
        logger.debug('number of nodes: {:d}'.format(num_nodes))
        logger.debug('number of edges: {:d}'.format(num_edges))
        logger.debug('number of blocks: {:d}'.format(num_blocks))
        logger.debug('-' * 5 + ' start iterating ' + '-' * 5)

    start_time = time.time()
    acc_proj_time = 0.
    func = BlockSumEMS(features=instance['features'], trade_off=trade_off, nodes_set=nodes_set, boundary_edges_dict=boundary_edges_dict, nodes_id_dict=nodes_id_dict)
    if logger:
        true_x = np.zeros(num_nodes)
        true_x[true_subgraph] = 1.
        true_x = np.array(true_x)
        true_obj_val, true_ems_val, true_penalty, _ = func.get_obj_val(true_x, boundary_edges_dict)
        logger.debug('ground truth, obj value: {:.5f}, sum ems value: {:.5f}, penalty: {:.5f}'.format(true_obj_val, true_ems_val, true_penalty))

    current_x = func.get_init_x_zeros() + 1e-6
    for iter in range(max_iter):
        if logger:
            logger.debug('iteration: {:d}'.format(iter))
        prev_x = np.copy(current_x)
        iter_time = time.time()
        iter_proj_time = 0.

        bx_array = np.zeros_like(current_x)
        omega_x_list = []
        for b in range(num_blocks):
            block_x = current_x[sorted(nodes_set[b])]
            boundary_xs_dict = get_boundary_xs(current_x, boundary_edges_dict[b], nodes_id_dict)
            feat = features[sorted(nodes_set[b])]
            grad_x = func.get_gradient(block_x, feat, boundary_xs_dict)

            block_x = block_x if iter > 0 else np.zeros_like(block_x, dtype=np.float64)

            normalized_grad = normalize_gradient(block_x, grad_x)

            edges = np.array(relabeled_edges_set[b])
            costs = np.ones(len(edges))
            start_proj_time = time.time()
            # g: number of connected component
            re_head = head_proj(edges=edges, weights=costs, x=normalized_grad, g=1, s=sparsity, budget=sparsity - 1., delta=1. / 169., max_iter=100, err_tol=1e-8, root=-1, pruning='strong', epsilon=1e-10, verbose=0)
            re_nodes, re_edges, p_x = re_head
            iter_proj_time += (time.time() - start_proj_time)
            gamma_x = set(re_nodes) # note, elements in block_gamma_x are all local (block) node id
            # omega_x_list.append(gamma_x)

            indicator_x = np.zeros(len(block_x))  # note, zeros, not
            indicator_x[list(gamma_x)] = 1.
            boundary_xs_dict = get_boundary_xs(current_x, boundary_edges_dict[b], nodes_id_dict)
            # grad_x = func.get_gradient(block_x, feat, boundary_xs_dict)
            bx_array[nodes_set[b]] = current_x[nodes_set[b]] + learning_rate * grad_x * indicator_x


        for b in range(num_blocks):
            bx = bx_array[nodes_set[b]]
            # get edges and edge weights of current block
            edges = np.array(relabeled_edges_set[b])
            costs = np.ones(len(edges))
            start_proj_time = time.time()
            re_tail = tail_proj(edges=edges, weights=costs, x=bx, g=1, s=sparsity, budget=sparsity - 1., nu=2.5, max_iter=100, err_tol=1e-8, root=-1, pruning='strong', verbose=0)
            re_nodes, re_edges, p_x = re_tail
            iter_proj_time += (time.time() - start_proj_time)
            psi_x = set(re_nodes)

            block_x = np.zeros_like(current_x[nodes_set[b]])
            block_x[list(psi_x)] = bx[list(psi_x)]
            normalized_block_x = normalize(block_x)
            current_x[nodes_set[b]] = normalized_block_x # constrain current block x in [0, 1]

        acc_proj_time += iter_proj_time

        if logger:
            obj_val, sum_ems_val, penalty, _ = func.get_obj_val(current_x, boundary_edges_dict)
            logger.debug('objective value: {:.5f}, sum ems value: {:.5f}, penalty: {:.5f}'.format(obj_val, sum_ems_val, penalty))
            logger.debug('iteration time: {:.5f}'.format(time.time() - iter_time))
            logger.debug('iter projection time: {:.5f}'.format(iter_proj_time))
            logger.debug('acc projection time: {:.5f}'.format(acc_proj_time))  # accumulative projection time
            logger.debug('-' * 10)

        diff_norm_x = np.linalg.norm(current_x - prev_x)
        if logger:
            logger.debug('difference {:.5f}'.format(diff_norm_x))
        if diff_norm_x < epsilon:
            break

    run_time = time.time() - start_time
    if logger:
        obj_val, sum_ems_val, penalty, _ = func.get_obj_val(current_x, boundary_edges_dict)
        logger.debug('objective value: {:.5f}, sum ems value: {:.5f}, penalty: {:.5f}'.format(obj_val, sum_ems_val, penalty))
        logger.debug('run time of whole algorithm: {:.5f}'.format(run_time))
        logger.debug('accumulative projection time: {:.5f}'.format(acc_proj_time))

    return current_x, run_time