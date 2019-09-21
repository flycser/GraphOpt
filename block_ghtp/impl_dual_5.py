#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : impl_dual_5
# @Date     : 09/05/2019 22:28:40
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     : ems density projection


from __future__ import print_function

import time

import numpy as np
import networkx as nx

from random import shuffle

from objs import DualEMS
from utils import normalize, normalize_gradient, evaluate_block

from sparse_learning.proj_algo import head_proj
from sparse_learning.proj_algo import tail_proj


def optimize(instance, sparsity, threshold, trade_off, learning_rate, max_iter, epsilon=1e-3, logger=None):

    first_graph = instance['first_graph']
    second_graph = instance['second_graph']
    true_subgraph = instance['true_subgraph']
    features = instance['weight']
    A = adj_matrix(second_graph) # get adjacent matrix of second graph, used for density projection

    first_graph_edges = np.array(first_graph.edges)
    first_graph_edge_weights = np.ones(first_graph.number_of_edges()) # edge weight, default 1

    print('number of nodes in first graph', first_graph.number_of_nodes())
    print('number of nodes in second graph', second_graph.number_of_nodes())

    if first_graph.number_of_nodes() != second_graph.number_of_nodes():
        raise ('error, wrong dual network input !!!')

    num_nodes = first_graph.number_of_nodes()
    num_edges_first_graph = first_graph.number_of_edges()
    num_edges_second_graph = second_graph.number_of_edges()

    if logger:
        # print some basic information
        logger.debug('-' * 5 + ' related info ' + '-' * 5)
        logger.debug('algorithm: graph block-structured GHTP')
        logger.debug('sparsity: {:d}'.format(sparsity))
        logger.debug('max iteration: {:d}'.format(max_iter))
        logger.debug('number of nodes: {:d}'.format(num_nodes))
        logger.debug('number of edges in first graph: {:d}'.format(num_edges_first_graph))
        logger.debug('number of edges in second graph: {:d}'.format(num_edges_second_graph))
        logger.debug('density of first graph: {:.5f}'.format(nx.density(first_graph)))
        logger.debug('density of second graph: {:.5f}'.format(nx.density(second_graph)))
        logger.debug('density of true subgraph in second graph: {:.5f}'.format(nx.density(nx.subgraph(second_graph, true_subgraph))))
        logger.debug('-' * 5 + ' start iterating ' + '-' * 5)

    start_time = time.time()
    acc_proj_time = 0.

    func = DualEMS(features, trade_off)
    if logger:
        print(sorted(true_subgraph))

        true_x = np.zeros(num_nodes)
        # print(type(true_subgraph))
        true_x[list(true_subgraph)] = 1.
        true_x = np.array(true_x)
        true_obj_val, x_ems_val, y_ems_val, penalty = func.get_obj_val(true_x, true_x)
        print('ground truth values: {}, {}, {}, {}'.format(true_obj_val, x_ems_val, y_ems_val, penalty))

    current_x, current_y = func.get_init_x_zeros() # are there some other better initialization methods?
    current_x += 1e-6 # start from zero, plus 1e-6 avoid divide by zero error
    current_y += 1e-6

    print('iteration start funval', func.get_obj_val(current_x, current_y))

    for iter in range(max_iter): # external iteration
        if logger:
            logger.debug('iteration: {:d}'.format(iter))

        prev_x, prev_y = np.copy(current_x), np.copy(current_y) # store previous vectors for early termination

        # handle first graph
        grad_x = func.get_gradient(current_x, current_y)
        iter_proj_time = 0.
        if iter == 0: # from all zero vector
            norm_grad_x = normalize_gradient(np.zeros_like(current_x), grad_x)
        else:
            norm_grad_x = normalize_gradient(current_x, grad_x)

        start_proj_time = time.time()
        # head projection for the connected constraint, so projection should be on first graph
        re_head = head_proj(edges=first_graph_edges, weights=first_graph_edge_weights, x=norm_grad_x, g=1, s=sparsity, budget=sparsity - 1., delta=1. / 169., max_iter=100, err_tol=1e-8, root=-1, pruning='strong', epsilon=1e-10, verbose=0)
        re_nodes, _, _ = re_head
        iter_proj_time += (time.time() - start_proj_time)
        print('head projection time for x: {:.5f}'.format(time.time() - start_proj_time))
        gamma_x = set(re_nodes)
        indicator_x = np.zeros(num_nodes)
        indicator_x[list(gamma_x)] = 1.
        # there is no differene between using grad_x and norm_grad_x, because indicator_x is got from norm_grad_x
        if iter == 0:
            tmp_x = np.zeros_like(current_x) + learning_rate * norm_grad_x * indicator_x # start from all zeros
        else:
            tmp_x = current_x + learning_rate * norm_grad_x * indicator_x

        omega_x = set([ind for ind, _ in enumerate(tmp_x) if not 0. == _])

        grad_y = func.get_gradient(current_y, current_x) # note, order
        # note, test not normalize
        if iter == 0:
            norm_grad_y = normalize_gradient(np.zeros_like(current_y), grad_y) # # note, is it necessary for density projection?
        else:
            norm_grad_y = normalize_gradient(current_y, grad_y)
            # norm_grad_y = grad_y # note !!!

        # note, should be positive for gradient, input for density projection should be positive
        # note, why baojian's code does not consider positive value, head projection
        abs_norm_grad_y = np.absolute(norm_grad_y) # take absolute value of gradient, since larger absolute value represent larger affection to objective function

        np.set_printoptions(linewidth=3000)

        # print(norm_grad_y)

        # lmbd_list = [0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.07, 0.07, 0.07, 0.08, 0.08, 0.08] # normalize
        lmbd_list = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005] # normalize
        # sparsity_list = [45, 50, 55, 45, 50, 55, 45, 50, 55, 55, 60, 65] # normalize
        sparsity_list = [50, 50, 50, 50, 55] # normalize
        lmbd_sparsity_list = zip(lmbd_list, sparsity_list)
        # sparsity_list = [50]
        print('start head projection for y')
        start_proj_time = time.time()
        # gamma_y = density_projection(second_graph, norm_grad_y, A, threshold, min_sparsity, max_sparsity, step_sparsity, normalize=False)
        # gamma_y = density_projection(second_graph, abs_norm_grad_y, A, threshold, lmbd_sparsity_list, normalize=True, true_subgraph=true_subgraph) # test not normalize, need new lambda sparsity list
        gamma_y = density_projection(second_graph, abs_norm_grad_y, A, threshold, lmbd_sparsity_list, normalize=False, true_subgraph=true_subgraph) # test not normalize, need new lambda sparsity list
        iter_proj_time += (time.time() - start_proj_time)
        print('head projection time for y: {:.5f}'.format(time.time() - start_proj_time))



        indicator_y = np.zeros(num_nodes)
        indicator_y[list(gamma_y)] = 1.
        if iter == 0:
            # tmp_y = np.zeros_like(current_y) + learning_rate * grad_y * indicator_y
            tmp_y = np.zeros_like(current_y) + learning_rate * norm_grad_y * indicator_y # todo, pls note that update gradient should be normalized gradient
        else:
            # tmp_y = current_y + learning_rate * grad_y * indicator_y
            tmp_y = current_y + learning_rate * norm_grad_y * indicator_y

        omega_y = set([ind for ind, _ in enumerate(tmp_y) if not 0. == _])

        print('omega_x', len(omega_x))
        print(sorted(list(omega_x)))

        print('omega_y', len(omega_y))
        print(sorted(list(omega_y)))

        print('intersect', len(omega_y & omega_x))
        print(sorted(list(omega_y & omega_x)))

        break

        print('solve argmax')
        start_max_time = time.time()
        bx, by = func.argmax_obj_with_proj(current_x, current_y, omega_x, omega_y)
        print('solve argmax time {:.5f}'.format(time.time() - start_max_time))

        # break

        start_proj_time = time.time()
        # tail projection on first graph
        re_tail = tail_proj(edges=first_graph_edges, weights=first_graph_edge_weights, x=bx, g=1, s=sparsity, budget=sparsity - 1., nu=2.5, max_iter=100, err_tol=1e-8, root=-1, pruning='strong', verbose=0)  # tail projection
        re_nodes, _, _ = re_tail
        iter_proj_time += time.time() - start_proj_time
        print('tail projection time for x: {:.5f}'.format(time.time() - start_proj_time))
        psi_x = set(re_nodes)

        current_x = np.zeros_like(current_x)
        current_x[list(psi_x)] = bx[list(psi_x)]
        current_x = normalize(current_x)

        lmbd_list = [0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.07, 0.07, 0.07, 0.08, 0.08, 0.08]
        # lmbd_list = [0.006, 0.08]
        sparsity_list = [45, 50, 55, 45, 50, 55, 45, 50, 55, 55, 60, 65]
        lmbd_sparsity_list = zip(lmbd_list, sparsity_list)

        start_proj_time = time.time()
        # psi_y = density_projection(second_graph, by, threshold, min_sparsity, max_sparsity, step_sparsity, normalize=False)
        psi_y = density_projection(second_graph, by, A, threshold, lmbd_sparsity_list, normalize=False)
        iter_proj_time += (time.time() - start_proj_time)

        print('tail projetion time for y: {:.5f}'.format(time.time() - start_proj_time))

        current_y = np.zeros_like(current_y)
        print('1', len(np.nonzero(by)[0]))
        print('by nonzero', sorted(list(np.nonzero(by)[0])))
        print('1v', len(np.nonzero(bx)[0]))
        print('2', len(psi_y))
        print('psi_y', sorted(list(psi_y)))
        print('2v', len(psi_x))
        current_y[list(psi_y)] = by[list(psi_y)]
        print('3', len(np.nonzero(current_y)[0]))
        print('3v', len(np.nonzero(current_x)[0]))
        current_y = normalize(current_y)
        print('4', len(np.nonzero(current_y)[0]))
        print('4v', len(np.nonzero(current_x)[0]))

        print('{} iteration funval'.format(iter), func.get_obj_val(current_x, current_y))

        acc_proj_time += iter_proj_time

        if logger:
            print('iter proj time: {:.5f}'.format(iter_proj_time))

        diff_norm = np.sqrt(np.linalg.norm(current_x - prev_x) ** 2 + np.linalg.norm(current_y - prev_y) ** 2)
        if logger:
            logger.debug('difference norm: {}'.format(diff_norm))

        if diff_norm < epsilon:
            break

    run_time = time.time() - start_time
    if logger:
        pass

    return current_x, current_y, run_time


# def density_projection(graph, weight, A, threshold, min_lmbd, max_lmbd, step_lmbd, min_sparsity, max_sparsity, step_sparsity, max_iter=10, normalize=False):
def density_projection(graph, weight, A, threshold, lmbd_sparsity_list, normalize=False, true_subgraph=None):

    if normalize:
        weight = (weight - np.min(weight)) / (np.max(weight) - np.min(weight))
    else:
        pass
    print('start density projection')

    density_subgraph_list = []
    print('density projection')
    # for lmbd in range(min_lmbd, max_lmbd, step_lmbd):
    for lmbd, sparsity in lmbd_sparsity_list:
    # for lmbd in lmbd_list:
        # for sparsity in range(min_sparsity, max_sparsity, step_sparsity):
        # for sparsity in sparsity_list:
        print('lambda, sparsity', lmbd, sparsity)
        x = proj_mp(graph, weight, A, sparsity, lmbd, max_iter=10, epsilon=1e-3) # max iteration

        print('performance', evaluate_block(true_subgraph, list(x)))
        density = nx.density(nx.subgraph(graph, x))

        print('density', density)
        if density > threshold:
            print(density, len(x), sorted(list(x)))
            density_subgraph_list.append((density, x))

    # find the best result
    print('candidate', len(density_subgraph_list))
    max_weight = 0.
    best_density_subgraph = None
    for density_subgraph in density_subgraph_list:
        tmp_weight = np.linalg.norm(weight[list(density_subgraph[1])])
        if tmp_weight > max_weight:
            best_density_subgraph = density_subgraph
            max_weight = tmp_weight

    print('end density projection: density {}, norm {}, size {}, set {}'.format(best_density_subgraph[0], max_weight, len(best_density_subgraph[1]), sorted(list(best_density_subgraph[1]))))

    return best_density_subgraph[1]


def adj_matrix(graph):
    """
    node must index from 0
    :param graph:
    :return:
    """
    A = np.zeros((graph.number_of_nodes(), graph.number_of_nodes()))
    for node_1, node_2 in graph.edges():
        A[node_1][node_2] = 1.
        A[node_2][node_1] = 1.

    return A

def proj_init_point(n):
    return np.zeros(n, dtype=np.float64)

def proj_mp(graph, weight, A, sparsity, lmbd, max_iter=10, epsilon=1e-3):
    current_x = proj_init_point(graph.number_of_nodes()) + 1e-6

    edges = np.array(graph.edges)
    edge_weights = np.ones(graph.number_of_edges())
    start_time = time.time()
    for i in range(max_iter):
        print('iter {}'.format(i))
        iter_time = time.time()
        gradient = proj_get_gradient(current_x, weight, A, lmbd)
        normalized_gradient = normalize_gradient(current_x, gradient)

        re_head = head_proj(edges=edges, weights=edge_weights, x=normalized_gradient, g=1, s=sparsity, budget=sparsity-1., delta=1. / 169., err_tol=1e-8, max_iter=100, root=-1, pruning='strong', epsilon=1e-10, verbose=0)
        re_nodes, _, _ = re_head
        gamma_x = set(re_nodes)
        print('gamma_x', len(gamma_x))
        # print(sorted(list(gamma_x)))
        if i == 0:
            supp_x = set()
        else:
            supp_x = set([ind for ind, _ in enumerate(current_x) if not 0. == _])
        omega_x = gamma_x | supp_x

        print('omega_x', len(omega_x))
        # print(sorted(list(omega_x)))

        # print(gradient[sorted(list(gamma_x))])
        # print(gradient[sorted(list(supp_x))])

        bx = proj_argmax(current_x, omega_x, weight, A, lmbd, max_iter=2000, learning_rate=0.01)

        re_tail = tail_proj(edges=edges, weights=edge_weights, x=bx, g=1, s=sparsity, budget=sparsity-1., nu=2.5, max_iter=100, err_tol=1e-8, root=-1, pruning='strong', verbose=0)
        re_nodes, _, _ = re_tail

        psi_x = set(re_nodes)
        prev_x = current_x
        current_x = np.zeros_like(current_x)
        current_x[list(psi_x)] = bx[list(psi_x)]

        print('psi_x', len(np.nonzero(current_x)[0]))

        diff_norm_x = np.linalg.norm(current_x - prev_x)
        func_val, xtw, dense_term = proj_get_func_val(current_x, weight, A, lmbd)

        print('iter {}, func val: {:.5f}, iter_time: {:.5f}, diff_norm: {:.5f}'.format(i, func_val, time.time() - iter_time, diff_norm_x))

        subgraph = set(np.nonzero(current_x)[0])
        print('subgraph density', nx.density(nx.subgraph(graph, subgraph)))

        if diff_norm_x <= epsilon:
            break

    run_time = time.time() - start_time
    func_val, xtw, dense_term = proj_get_func_val(current_x, weight, A, lmbd)

    print('final function value: {:.5f}'.format(func_val))
    print('run time of whole algorithm: {:.5f}'.format(run_time))

    subgraph = set(np.nonzero(current_x)[0])

    return subgraph


def proj_get_func_val(x, w, A, lmbd):

    func_val = 0.
    xt1 = np.sum(x)
    xtw = w.T.dot(x)
    func_val += lmbd * xtw / np.sqrt(xt1)

    Atx = np.dot(A, x)
    dense_term = np.dot(Atx/xt1, x)
    func_val += dense_term

    return func_val, xtw, dense_term


def proj_argmax(x, omega, w, A, lmbd, max_iter=2000, learning_rate=0.01, epsilon=1e-3):
    indicator = np.zeros_like(x)
    indicator[list(omega)] = 1.

    current_x = np.copy(x)
    for i in range(max_iter):
        prev_x = np.copy(current_x)
        gradient = proj_get_gradient(x, w, A, lmbd)

        current_x = proj_maximizer(current_x, gradient, indicator, learning_rate, bound=5)

        diff_norm_x = np.linalg.norm(current_x - prev_x)

        if diff_norm_x <= epsilon:
            break

    return current_x

def proj_maximizer(x, gradient, indicator, learning_rate, bound=5):

    normalized_x = (x + learning_rate * gradient) * indicator
    sorted_indices = np.argsort(normalized_x)[::-1]

    normalized_x[normalized_x <= 0.] = 0.
    num_non_psi = len(np.where(normalized_x == 0.)[0])
    normalized_x[normalized_x > 1.] = 1.
    if num_non_psi == len(x):
        print("siga-1 is too large and all values in the gradient are non-positive")
        for i in range(bound):
            normalized_x[sorted_indices[i]] = 1.0

    return normalized_x


def proj_get_gradient(x, w, A, lmbd):
    if len(x) != len(w):
        print('Error: Invalid parameter ...')
        return None
    elif np.sum(x) == 0:
        print('Error: x vector all zeros ...')
        return None

    non_zero_count = 0.
    for i in range(len(x)):
        if x[i] > 0. and x[i] <= 1.:
            non_zero_count += 1.

    xt1 = np.sum(x)
    xtw = np.dot(x, w)
    Ax = np.dot(A, x)
    xtAx = np.dot(x, Ax)
    gradient = lmbd * (w / xt1 - .5 * xtw / np.power(xt1, 1.5)) + (2 * Ax * xt1 - xtAx) / (xt1 ** 2) # note, first term, ems gradient

    return gradient




if __name__ == '__main__':
    x = np.array([1, 2, 3])
    A = np.array([[1, 1, 2], [3, 3, 1], [2, 2, 1]])
    Ax = np.dot(A, x)
    print(Ax)
    xtAx = np.dot(x, Ax)
    print(xtAx)
