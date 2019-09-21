#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : impl_dual_2
# @Date     : 08/02/2019 17:00:20
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


from __future__ import print_function

import time

import numpy as np
import networkx as nx

from random import shuffle

from objs import DualEMS
from utils import relabel_nodes, normalize, normalize_gradient

from multiprocessing import Pool, Process, Manager

from sparse_learning.proj_algo import head_proj
from sparse_learning.proj_algo import tail_proj


def optimize(instance, sparsity, threshold, trade_off, learning_rate, max_iter, epsilon=1e-3, logger=None):

    first_graph = instance['first_graph']
    second_graph = instance['second_graph']
    # true_subgraph = instance['true_subgraph']
    true_subgraph = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] # note
    features = instance['weight']

    first_graph_edges = np.array(first_graph.edges)
    # second_graph_edges = np.array(second_graph.edges)
    first_graph_edge_weights = np.ones(first_graph.number_of_edges())
    # second_graph_edge_weights = np.ones(second_graph.number_of_edges())

    print(first_graph.number_of_nodes())
    print(second_graph.number_of_nodes())

    if first_graph.number_of_nodes() != second_graph.number_of_nodes():
        raise('error, wrong dual network input !!!')

    num_nodes = first_graph.number_of_nodes()
    num_edges_first_graph = first_graph.number_of_edges()
    num_edges_second_graph = second_graph.number_of_edges()

    if logger:
        logger.debug('-' * 5 + ' related info ' + '-' * 5)
        logger.debug('algorithm: graph block-structured GHTP')
        logger.debug('sparsity: {:d}'.format(sparsity))
        logger.debug('max iteration: {:d}'.format(max_iter))
        logger.debug('number of nodes: {:d}'.format(num_nodes))
        logger.debug('number of edges in first graph: {:d}'.format(num_edges_first_graph))
        logger.debug('number of edges in second graph: {:d}'.format(num_edges_second_graph))
        logger.debug('density of first graph: {:.5f}'.format(nx.density(first_graph)))
        logger.debug('density of second graph: {:.5f}'.format(nx.density(second_graph)))
        logger.debug('-' * 5 + ' start iterating ' + '-' * 5)

    # calculate shortest path lengths for all node pairs, which will be used in density projection
    start_time = time.time()
    shortest_path_lengths = {}
    for node in second_graph.nodes():
        x = nx.shortest_path_length(second_graph, source=node)
        shortest_path_lengths[node] = x
    print('shortest_path time: {:.5f}'.format(time.time() - start_time))

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

    # initialize node coefficients
    current_x, current_y = func.get_init_x_zeros()
    current_x += 1e-6 # from not zeros but close to zero, avoid error divided by zero
    current_y += 1e-6

    print('iteration start funval', func.get_obj_val(current_x, current_y))

    # start optimization
    for iter in range(max_iter):
        if logger:
            logger.debug('iteration: {:d}'.format(iter))
        prev_x, prev_y = np.copy(current_x), np.copy(current_y)

        # handle first graph
        grad_x = func.get_gradient(current_x, current_y)
        iter_proj_time = 0.
        # iter_time = time.time()
        if iter == 0:
            norm_grad_x = normalize_gradient(np.zeros_like(current_x), grad_x)
        else:
            norm_grad_x = normalize_gradient(current_x, grad_x)


        # head projection
        start_proj_time = time.time()
        re_head = head_proj(edges=first_graph_edges, weights=first_graph_edge_weights, x=norm_grad_x, g=1, s=sparsity, budget=sparsity - 1., delta=1. / 169., max_iter=100, err_tol=1e-8, root=-1, pruning='strong', epsilon=1e-10, verbose=0)  # head projection
        re_nodes, _, _ = re_head
        iter_proj_time += (time.time() - start_proj_time)
        print('head projection time for x: {:.5f}'.format(time.time() - start_proj_time))
        gamma_x = set(re_nodes)
        indicator_x = np.zeros(num_nodes)
        indicator_x[list(gamma_x)] = 1.
        if iter == 0:
            tmp_x = np.zeros_like(current_x) + learning_rate * grad_x * indicator_x  # note, not update current variables, only use the intermediate results
        else:
            tmp_x = current_x + learning_rate * grad_x * indicator_x

        omega_x = set([ind for ind, _ in enumerate(tmp_x) if not 0. == _])

        # handle second graph
        grad_y = func.get_gradient(current_y, current_x)
        # note, test not normalize
        if iter == 0:
            norm_grad_y = normalize_gradient(np.zeros_like(current_y), grad_y)
        else:
            norm_grad_y = normalize_gradient(current_y, grad_y)
            # norm_grad_y = grad_y # note, test !!!

        # note, should positive, eventallpairs algorithm require that node weights should be positive
        norm_grad_y = np.absolute(norm_grad_y)

        start = 5
        ratio = 5
        steps = 8
        progression = [start * ratio ** i for i in range(steps)] # generate geometric sequence, enumerate as lambda parameter in eventallpairs algorithm
        print('lmbd progression', progression)
        start_proj_time = time.time()
        print(norm_grad_y)
        gamma_y = dense_projection(second_graph, norm_grad_y, threshold, progression, shortest_path_lengths, normalize=True, sort=False) # normalize True
        iter_proj_time += (time.time() - start_proj_time)
        print('head projection time for y: {:.5f}'.format(time.time() - start_proj_time))

        indicator_y = np.zeros(num_nodes)
        indicator_y[list(gamma_y)] = 1.
        if iter == 0:
            tmp_y = np.zeros_like(current_y) + learning_rate * grad_y * indicator_y # note, not update current variables, only use the intermediate results
        else:
            tmp_y = current_y + learning_rate * grad_y * indicator_y

        omega_y = set([ind for ind, _ in enumerate(tmp_y) if not 0. == _])

        print('solve argmax')
        # solve argmax
        start_max_time = time.time()
        bx, by = func.argmax_obj_with_proj(current_x, current_y, omega_x, omega_y)
        print('solve argmax time: {:.5f}'.format(time.time() - start_max_time))

        # tail projection for the first graph
        start_proj_time = time.time()
        re_tail = tail_proj(edges=first_graph_edges, weights=first_graph_edge_weights, x=bx, g=1, s=sparsity, budget=sparsity - 1., nu=2.5, max_iter=100, err_tol=1e-8, root=-1, pruning='strong', verbose=0)  # tail projection
        re_nodes, _, _ = re_tail
        iter_proj_time += time.time() - start_proj_time
        print('tail projection time for x: {:.5f}'.format(time.time() - start_proj_time))
        psi_x = set(re_nodes)

        current_x = np.zeros_like(current_x)
        current_x[list(psi_x)] = bx[list(psi_x)]
        current_x = normalize(current_x)  # note, constrain current_x in [0, 1], is this step necessary

        # print(by) # note, by in [0, 1], so we should change lmbd range

        # start = 5
        # ratio = 10
        progression = [start * ratio ** i for i in range(steps)]
        # tail projection for the second graph
        start_proj_time = time.time()
        psi_y = dense_projection(second_graph, by, threshold, progression, shortest_path_lengths, normalize=False, sort=False) # note, normalize ???
        iter_proj_time += (time.time() - start_proj_time)

        print('tail projection time for y: {:.5f}'.format(time.time() - start_proj_time))

        current_y = np.zeros_like(current_y)
        # current_y[list(psi_y)] = bx[list(psi_y)] # note, error, but resutls are good
        current_y[list(psi_y)] = by[list(psi_y)]
        current_y = normalize(current_y)  # constrain current_y in [0, 1]

        print('{} iteration funval'.format(iter), func.get_obj_val(current_x, current_y))

        acc_proj_time += iter_proj_time

        if logger:
            print('iter proj time: {:.5f}'.format(iter_proj_time))

        diff_norm = np.sqrt(np.linalg.norm(current_x - prev_x) ** 2 + np.linalg.norm(current_y - prev_y) ** 2)
        if logger:
            logger.debug('difference norm: {}'.format(diff_norm))

            # raw_pred_subgraph_x = np.nonzero(current_x)[0]
            #
            # prec, rec, fm, iou = evaluate_block(instance['true_subgraph'], raw_pred_subgraph_x)
            #
            # logger.debug('-' * 5 + ' performance of x prediction ' + '-' * 5)
            # logger.debug('precision: {:.5f}'.format(prec))
            # logger.debug('recall   : {:.5f}'.format(rec))
            # logger.debug('f-measure: {:.5f}'.format(fm))
            # logger.debug('iou      : {:.5f}'.format(iou))
            #
            # raw_pred_subgraph_y = np.nonzero(current_y)[0]
            #
            # prec, rec, fm, iou = evaluate_block(instance['true_subgraph'], raw_pred_subgraph_y)
            #
            # logger.debug('-' * 5 + ' performance of y prediction ' + '-' * 5)
            # logger.debug('precision: {:.5f}'.format(prec))
            # logger.debug('recall   : {:.5f}'.format(rec))
            # logger.debug('f-measure: {:.5f}'.format(fm))
            # logger.debug('iou      : {:.5f}'.format(iou))


        if diff_norm < epsilon:
            break

    run_time = time.time() - start_time
    if logger:
        pass

    return current_x, current_y, run_time


def dense_projection(graph, weight, threshold, progression, shortest_path_lengths, normalize=False, sort=False, times=2):

    print('dense projection')

    manager = Manager()
    return_dict = manager.dict()
    jobs = []

    for lmbd in progression:
        # for each lmbd, test 5 times, since this algorithm is a randomized algorithm, we should run many times and find a better one
        # for i in range(times):
        #     print('test {} {}'.format(lmbd, i))
        #     X = BFNS(graph, weight, lmbd, normalize=normalize, sort=sort, verbose=False)
        #     density = nx.density(nx.subgraph(graph, X))
        #     if density > threshold:
        #         print(density, len(X), sorted(list(X)))
        #         density_subgraph_list.append((density, X))

        verbose = True

        para = graph, weight, lmbd, shortest_path_lengths, normalize, sort, verbose

        process = Process(target=BFNS_worker, args=(para, return_dict))
        jobs.append(process)
        process.start()

    for proc in jobs:
        proc.join()

    # how do we select the best candidate subgraph?
    # strategy 1, find the density larger than threshold, and maximize weights in sets
    # e.g. \|weight_{X}\|_2 maximized
    max_weight = 0
    best_density_subgraph = None
    for b in return_dict.keys():
        density_subgraph = return_dict[b]
        tmp_weight = np.linalg.norm(weight[list(density_subgraph)]) # note, test another metric, max obj score
        density = nx.density(nx.subgraph(graph, return_dict[b]))
        if tmp_weight > max_weight and density > threshold:
            best_density_subgraph = (density, density_subgraph)
            max_weight = tmp_weight

    print('density projection: density {}, norm {}, size {}, set {}'.format(best_density_subgraph[0], max_weight, len(best_density_subgraph[1]), sorted(list(best_density_subgraph[1]))))

    return best_density_subgraph[1]


def BFNS_worker(para, return_dict):
    graph, weight, lmbd, shortest_path_lengths, normalize, sort, verbose = para

    # normalize weight
    if normalize:
        weight = (weight - np.min(weight)) / (np.max(weight) - np.min(weight))
    else:
        weight = weight - np.min(weight)

    if sort:
        weight[::-1].sort()

    nodes = [node for node in graph.nodes()]
    shuffle(nodes)

    X = set()  # start from empty set
    Y = set([node for node in graph.nodes()])  # start from entire node set

    start_time = time.time()

    for i in range(graph.number_of_nodes()):
        # how to select next node? randomly? shuffle node list firstly
        u = nodes[i]
        shortest_lengths_from_u = nx.shortest_path_length(graph, source=u)

        #   Q_AP(X\cup u) - Q_AP(X)
        # = I(X\cup u) + D_AP(V) - D_AP(X\cup u) - I(X) - D_AP(V) + D_AP(X)
        # = I(X) + I(u) + D_AP(V) - D_AP(X\cup u) - I(X) - D_AP(V) + D_AP(X)
        # = I(u) + D(X) - D_AP(X\cup u)
        Dap_a = 0
        for node in X:
            if node not in shortest_path_lengths[u].keys():  # note, handle not connected nodes
                Dap_a -= max(shortest_path_lengths[u].values())
            else:
                Dap_a -= shortest_path_lengths[u][node]
        a = lmbd * weight[u] + Dap_a

        #   Q_AP(Y\u) - Q_AP(Y)
        # = I(Y\u) + D_AP(V) - D_AP(Y\u) - I(Y) - D_AP(V) + D_AP(Y)
        # = I(Y) - I(u) + D_AP(V) - D_AP(Y\u) - I(Y) - D_AP(V) + D_AP(Y)
        # = -I(u) + D_AP(Y) - D_AP(Y\u)
        Dap_b = 0
        for node in Y:
            if node not in shortest_path_lengths[u].keys():  # note, handle not connected nodes
                Dap_b += max(shortest_path_lengths[u].values())
            else:
                Dap_b += shortest_path_lengths[u][node]

        b = - lmbd * weight[u] + Dap_b

        a = a if a > 0 else 0
        b = b if b > 0 else 0

        if a == b == 0:
            a = b = 0.5

        if np.random.uniform() < a / (a + b):
            X.add(u)
            # if verbose:
            #     print('add node {:d} to X, margin gain: {:.5f}'.format(u, a))
        else:
            Y.remove(u)
            # if verbose:
            #     print('remove node {:d} from Y, margin gain: {:.5f}'.format(u, b))

    if verbose:
        print('{} run time: {:.5f} sec.'.format(lmbd, time.time() - start_time))

    # print(sorted(list(X)))
    # print(sorted(list(Y)))

    return_dict[lmbd] = X



def greedAP(graph, weight, lmbd, normalize=True, sort=False, verbose=True):

    # normalize weight
    if normalize:
        weight = (weight - np.min(weight)) / (np.max(weight) - np.min(weight))
    else:
        weight = weight - np.min(weight)  # note, positive

    if sort:
        weight[::-1].sort()

    nodes = [node for node in graph.nodes()]
    shuffle(nodes)

    X = set()  # start from empty set

    start_time = time.time()
    # add node util marginal gain is negative
    X.add(nodes[0])
    i = 1
    while True:
        if i >= len(nodes):
            break

        u = nodes[i]
        shortest_lengths_from_u = nx.shortest_path_length(graph, source=u)
        Dap = 0
        for node in X:
            Dap -= shortest_lengths_from_u[node]

        margin_gain = lmbd * weight[u] + Dap

        if margin_gain < 0:
            if verbose:
                print('stop at node {:d}, margin gain: {:.5f}'.format(u, margin_gain))
            break
        else:
            X.add(u)
            if verbose:
                print('add node {:d}, node weight {:.2f}, margin gain: {:.5f}'.format(u, weight[u], margin_gain))

        i += 1

    if verbose:
        print('run time: {:.5f} sec.'.format(time.time() - start_time))

    return X


def BFNS(graph, weight, lmbd, normalize=True, sort=True, verbose=True):
    # normalize weight
    if normalize:
        weight = (weight - np.min(weight)) / (np.max(weight) - np.min(weight))
    else:
        weight = weight - np.min(weight)

    if sort:
        weight[::-1].sort()

    nodes = [node for node in graph.nodes()]
    shuffle(nodes)

    X = set()  # start from empty set
    Y = set([node for node in graph.nodes()])  # start from entire node set

    start_time = time.time()

    for i in range(graph.number_of_nodes()):
        # how to select next node? randomly? shuffle node list firstly
        u = nodes[i]
        shortest_lengths_from_u = nx.shortest_path_length(graph, source=u)

        #   Q_AP(X\cup u) - Q_AP(X)
        # = I(X\cup u) + D_AP(V) - D_AP(X\cup u) - I(X) - D_AP(V) + D_AP(X)
        # = I(X) + I(u) + D_AP(V) - D_AP(X\cup u) - I(X) - D_AP(V) + D_AP(X)
        # = I(u) + D(X) - D_AP(X\cup u)
        Dap_a = 0
        for node in X:
            if node not in shortest_lengths_from_u.keys(): # note, handle not connected nodes
                Dap_a -= max(shortest_lengths_from_u.values())
            else:
                Dap_a -= shortest_lengths_from_u[node]
        a = lmbd * weight[u] + Dap_a

        #   Q_AP(Y\u) - Q_AP(Y)
        # = I(Y\u) + D_AP(V) - D_AP(Y\u) - I(Y) - D_AP(V) + D_AP(Y)
        # = I(Y) - I(u) + D_AP(V) - D_AP(Y\u) - I(Y) - D_AP(V) + D_AP(Y)
        # = -I(u) + D_AP(Y) - D_AP(Y\u)
        Dap_b = 0
        for node in Y:
            if node not in shortest_lengths_from_u.keys(): # note, handle not connected nodes
                Dap_b += max(shortest_lengths_from_u.values())
            else:
                Dap_b += shortest_lengths_from_u[node]

        b = - lmbd * weight[u] + Dap_b

        a = a if a > 0 else 0
        b = b if b > 0 else 0

        if a == b == 0:
            a = b = 0.5

        if np.random.uniform() < a / (a + b):
            X.add(u)
            if verbose:
                print('add node {:d} to X, margin gain: {:.5f}'.format(u, a))
        else:
            Y.remove(u)
            if verbose:
                print('remove node {:d} from Y, margin gain: {:.5f}'.format(u, b))

    if verbose:
        print('run time: {:.5f} sec.'.format(time.time() - start_time))

    # print(sorted(list(X)))
    # print(sorted(list(Y)))

    return X