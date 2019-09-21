#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : max_dense_subgraph_proj_3
# @Date     : 08/18/2019 16:26:41
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     : uniform distribution generate dataset


from __future__ import print_function

import time
import pickle

import numpy as np
import networkx as nx

from sparse_learning.proj_algo import head_proj
from sparse_learning.proj_algo import tail_proj


def get_func_val(x, w, A, lmbd):
    func_val = 0.

    xt1 = np.sum(x)
    xtw = w.T.dot(x)
    func_val += lmbd * xtw

    Atx = np.dot(A, x)
    dense_term = np.dot(Atx/xt1, x)
    func_val += dense_term

    return func_val, xtw, dense_term


def normalize_gradient(x, gradient):
    if len(gradient) == 0 or len(x) == 0:
        print('Error: gradient/x is None ...')

    normalized_gradient = np.zeros_like(x)
    for i in range(len(gradient)):
        if gradient[i] > 0. and x[i] == 1.:
            normalized_gradient[i] = 0.
        elif gradient[i] < 0. and x[i] == 0.:
            normalized_gradient[i] = 0.
        else:
            normalized_gradient[i] = gradient[i]


    return normalized_gradient


def get_gradient(x, w, A, lmbd):
    if len(x) != len(w):
        print('Error: Invalid parameter ...')
        return None
    elif np.sum(x) == 0:
        print('Error: x vector all zeros ...')
        return None

    non_zero_count = 0.
    for i in range(len(x)):
        if x[i] > 0. and x[i] < 1.:
            non_zero_count += 1.

    xt1 = np.sum(x)
    Ax = np.dot(A, x)
    xtAx = np.dot(x, Ax)
    gradient = lmbd * w + (2 * Ax * xt1 - xtAx) / (xt1 ** 2)

    return gradient


def argmax(x, omega, w, A, lmbd, max_iter=2000, learning_rate=1., epsilon=1e-3):
    indicator = np.zeros_like(x)
    indicator[list(omega)] = 1.

    current_x = np.copy(x)
    for i in range(max_iter):
        prev_x = np.copy(current_x)
        gradient = get_gradient(x, w, A, lmbd)

        current_x = maximizer(current_x, gradient, indicator, learning_rate, bound=5)

        diff_norm_x = np.linalg.norm(current_x - prev_x)

        if diff_norm_x <= epsilon:
            break

    return current_x

def maximizer(x, gradient, indicator, learning_rate, bound=5):

    normalized_x = (x + learning_rate * gradient) * indicator
    sorted_indices = np.argsort(normalized_x)[::-1]

    normalized_x[normalized_x <= 0.0] = 0.0
    num_non_psi = len(np.where(normalized_x == 0.0)[0]) # todo, api where returns a ndarray
    normalized_x[normalized_x > 1.0] = 1.0
    if num_non_psi == len(x):
        print("siga-1 is too large and all values in the gradient are non-positive")
        for i in range(bound):
            normalized_x[sorted_indices[i]] = 1.0

    return normalized_x


def init_point(n):

    return np.zeros(n, dtype=np.float64)


def adj_matrix(graph):
    A = np.zeros((graph.number_of_nodes(), graph.number_of_nodes()))
    for node_1, node_2 in graph.edges():
        A[node_1][node_2] = 1.
        A[node_2][node_1] = 1.

    return A


def optimize(graph, w, A, sparsity, lmbd, max_iter, epsilon=1e-3):

    current_x = init_point(graph.number_of_nodes()) + 1e-6

    edges = np.array(graph.edges)
    edge_weights = np.ones(graph.number_of_edges())
    start_time = time.time()
    for i in range(max_iter):
        print('iter {}'.format(i))
        iter_time = time.time()
        gradient = get_gradient(current_x, w, A, lmbd)
        normalized_gradient = normalize_gradient(current_x, gradient)
        # normalized_gradient = gradient

        re_head = head_proj(edges=edges, weights=edge_weights, x=normalized_gradient, g=1, s=sparsity, budget=sparsity-1., delta=1. / 169., err_tol=1e-8, max_iter=100, root=-1, pruning='strong', epsilon=1e-10, verbose=0)
        re_nodes, _, _ = re_head
        gamma_x = set(re_nodes)
        if i == 0:
            supp_x = set()
        else:
            supp_x = set([ind for ind, _ in enumerate(current_x) if not 0. == _])
        omega_x = gamma_x | supp_x

        print('omegax', len(omega_x))
        # print(sorted(omega_x))

        bx = argmax(current_x, omega_x, w, A, lmbd, max_iter=2000, learning_rate=0.01)

        # print('nonzero', [bx[node] for node in sorted(list(np.nonzero(bx)[0]))])

        re_tail = tail_proj(edges=edges, weights=edge_weights, x=bx, g=1, s=sparsity, budget=sparsity-1., nu=2.5, max_iter=100, err_tol=1e-8, root=-1, pruning='strong', verbose=0)
        re_nodes, _, _ = re_tail
        # print(re_nodes)

        psi_x = set(re_nodes)
        prev_x = current_x
        current_x = np.zeros_like(current_x)
        current_x[list(psi_x)] = bx[list(psi_x)]

        print('psix', len(np.nonzero(current_x)[0]))

        diff_norm_x = np.linalg.norm(current_x - prev_x)
        func_val, xtw, dense_term = get_func_val(current_x, w, A, lmbd)

        print('iter {}, func val: {:.5f}, iter_time: {:.5f}, diff_norm: {:.5f}'.format(i, func_val, time.time() - iter_time, diff_norm_x))

        subgraph = set(np.nonzero(current_x)[0])
        print('subgraph density', nx.density(nx.subgraph(graph, subgraph)))

        if diff_norm_x < epsilon:
            break

    run_time = time.time() - start_time
    func_val, xtw, dense_term = get_func_val(current_x, w, A, lmbd)
    print('final function value: {:.5f}'.format(func_val))
    print('run time of whole algorithm: {:.5f}'.format(run_time))

    return current_x


def evaluate(subgraph, true_subgraph):
    subgraph, true_subgraph = set(subgraph), set(true_subgraph)

    prec = rec = fm = iou = 0.

    intersection = true_subgraph & subgraph
    union = true_subgraph | subgraph

    if not 0. == len(subgraph):
        prec = len(intersection) / float(len(subgraph))

    if not 0. == len(true_subgraph):
        rec = len(intersection) / float(len(true_subgraph))

    if prec + rec > 0.:
        fm = (2. * prec * rec) / (prec + rec)

    if union:
        iou = len(intersection) / float(len(union))

    return prec, rec, fm, iou


def test():
    # mu = 5
    # fn = '/network/rit/lab/ceashpc/share_data/GraphOpt/DCS/dense/syn/dataset_mu{}.pkl'.format(mu)

    elevate_val = 1

    fn = '/network/rit/lab/ceashpc/share_data/GraphOpt/DCS/dense/syn/dataset_ele{}.pkl'.format(elevate_val)
    with open(fn, 'rb') as rfile:
        dataset = pickle.load(rfile)

    # normalize = False
    normalize = True
    print('normalize={}, elevate={}'.format(normalize, elevate_val))
    for lmbd in [0.001]:
    # for lmbd in [0., 0.0005, 0.001, 0.005, 0.01, 0.05, 0.07]: # for elevate 1, ratio 2
    # for lmbd in [0., 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]: # for elevate 2, ratio 2
    # for lmbd in [0.0, 0.0005, 0.001, 0.005, 0.01, 0.02]: # for elevate 5, 10, ratio 2.
    # for lmbd in [0.0, 0.0005, 0.001, 0.005, 0.01, 0.02]: # for elevate 5, 10, ratio 4.
    # for lmbd in [0.0, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03]: # for elevate 5, 10, ratio 3.
    # for lmbd in [0.0, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07]: # for elevate 5, 10, ratio 1.
    # for lmbd in [0.0, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07]: # for elevate 2, ratio 1.
    # for lmbd in [0.0, 0.0005, 0.001, 0.005, 0.01, 0.02]: # for elevate 2, ratio 3.
    # for lmbd in [0.0, 0.0005, 0.001, 0.005, 0.01]: # for elevate 2, ratio 4.
    # for lmbd in [0.0, 0.0005, 0.001, 0.005, 0.01]: # for elevate 1, ratio 4.
    # for lmbd in [0.0, 0.0005, 0.001, 0.005, 0.01, 0.02]: # for elevate 1, ratio 3.
    # for lmbd in [0.0, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.07, 0.1]: # for elevate 1, ratio 1.
        results = []

        # for ratio in np.arange(2., 3., 1.):
        # for ratio in np.arange(1., 2., 1.):
        # for ratio in np.arange(4., 5., 1.):
        for ratio in np.arange(3., 4., 1.):
            for i, instance in enumerate(dataset):

                graph, weight, true_subgraph = instance['graph'], instance['weight'], instance['true_subgraph']
                A = adj_matrix(graph)
                sparsity = int(len(true_subgraph) / ratio)
                print('lambda={}, ratio={}, instance={}'.format(lmbd, ratio, i))
                if normalize:
                    weight = (weight - np.min(weight)) / (np.max(weight) - np.min(weight))

                x = optimize(graph, weight, A, sparsity=sparsity, lmbd=lmbd, max_iter=20)
                subgraph = set(np.nonzero(x)[0])

                prec, rec, fm, iou = evaluate(subgraph, true_subgraph)

                sum_weight_subgraph = np.sum(weight[list(subgraph)])

                subgraph = nx.subgraph(graph, subgraph)


                print('performance', prec, rec, fm, iou, nx.density(graph), nx.density(subgraph), subgraph.number_of_nodes(), subgraph.number_of_edges(), sum_weight_subgraph)

                results.append((prec, rec, fm, iou, nx.density(graph), nx.density(subgraph), subgraph.number_of_nodes(), subgraph.number_of_edges(), sum_weight_subgraph))

                # break
            # break
        # break

        results = np.array(results)
        print(results)
        np.set_printoptions(linewidth=1000)
        print('mean', np.array2string(np.mean(results, axis=0), separator='\t'))
        print('std', np.array2string(np.std(results, axis=0), separator='\t'))


def extract_results():
    fn = '/network/rit/lab/ceashpc/fjie/projects/GraphOpt/references/dcs/slurm-242222.out'


    results = {}

    with open(fn) as rfile:
        while True:
            line = rfile.readline()
            if not line:
                break

            if line.startswith('lambda'):
                terms = line.strip().split(',')
                lmbd = str(terms[0].split('=')[1])
                ratio = float(terms[1].split('=')[1])
                instance = int(terms[2].split('=')[1])
                while True:
                    line = rfile.readline()
                    if line.startswith('performance'):
                        terms = line.strip().split()
                        prec, rec, fm, iou, graph_density, subgraph_density, num_nodes, num_edges = float(terms[1]), float(terms[2]), float(terms[3]), float(terms[4]), float(terms[5]), float(terms[6]), int(terms[7]), int(terms[8])
                        break

                if lmbd not in results:
                    results[lmbd] = {}

                if instance not in results[lmbd]:
                    results[lmbd][instance] = {}

                if ratio not in results[lmbd][instance]:
                    results[lmbd][instance][ratio] = (prec, rec, fm, iou, graph_density, subgraph_density, num_nodes, num_edges)
                else:
                    results[lmbd][instance][ratio] = (prec, rec, fm, iou, graph_density, subgraph_density, num_nodes, num_edges)

    # print(results.keys())
    # for k in results:
    #     print(results[k].keys())

    # for lmbd in [0.0, 0.0005, 0.001, 0.005, 0.01, 0.02]: # elevate 10, 5
    for lmbd in [0., 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]: # elevate 2
        lmbd = '{}'.format(lmbd)
        for instance in range(10):
            for ratio in sorted(results[lmbd][instance]):
                print(lmbd, instance, ratio, end='\t')
                prec, rec, fm, iou, graph_density, subgraph_density, num_nodes, num_edges = results[lmbd][instance][ratio]
                print('\t'.join((str(round(prec, 4)), str(round(rec, 4)), str(round(fm, 4)), str(round(iou, 4)), str(round(graph_density, 4)), str(round(subgraph_density, 4)), str(num_nodes), str(num_edges))), end='\t')

            print()

        print()


if __name__ == '__main__':

    pass

    test()

    # extract_results()

