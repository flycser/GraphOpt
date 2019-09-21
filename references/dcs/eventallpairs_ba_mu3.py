#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : eventallpairs_ba_mu3
# @Date     : 09/04/2019 09:11:13
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


from __future__ import print_function

import os
import sys
import time
import pickle

import numpy as np
import networkx as nx

from random import shuffle


def BFNS(graph, weight, lmbd, normalize=True, sort=False, verbose=True):
    # normalize weight, or make weight positive
    if normalize:
        # pass
        weight = (weight - np.min(weight)) / (np.max(weight) - np.min(weight))
    else:
        weight = weight - np.min(weight)
        # pass

    # consider each node by its weight descendingly
    if sort:
        # weight[::-1].sort()
        pass


    nodes = [node for node in graph.nodes()]
    shuffle(nodes) # randomly

    X = set() # start from empty set
    Y = set([node for node in graph.nodes()]) # start from entire node set

    start_time = time.time()

    for i in range(graph.number_of_nodes()):
        # how to select next node? randomly? shuffle node list firstly
        u = nodes[i]
        shortest_lengths_from_u = nx.shortest_path_length(graph, source=u)

        #   Q_AP(X\cup u) - Q_AP(X)
        # = I(X\cup u) + D_AP(V) - D_AP(X\cup u) - I(X) - D_AP(V) + D_AP(X)
        # = I(X) + I(u) + D_AP(V) - D_AP(X\cup u) - I(X) - D_AP(V) + D_AP(X)
        # = I(u) + D_AP(X) - D_AP(X\cup u)
        Dap_a = 0
        for node in X:
            Dap_a -= shortest_lengths_from_u[node]
        a = lmbd * weight[u] + Dap_a

        #   Q_AP(Y\u) - Q_AP(Y)
        # = I(Y\u) + D_AP(V) - D_AP(Y\u) - I(Y) - D_AP(V) + D_AP(Y)
        # = I(Y) - I(u) + D_AP(V) - D_AP(Y\u) - I(Y) - D_AP(V) + D_AP(Y)
        # = -I(u) + D_AP(Y) - D_AP(Y\u)
        Dap_b = 0
        for node in Y:
            Dap_b += shortest_lengths_from_u[node]

        b = - lmbd * weight[u] + Dap_b

        a = a if a > 0 else 0
        b = b if b > 0 else 0

        if a == b == 0:
            a = b = 0.5

        if np.random.uniform() < a / (a+b):
            X.add(u)
            if verbose:
                print('add node {:d} to X, margin gain: {:.5f}'.format(u, a))
        else:
            Y.remove(u)
            if verbose:
                print('remove node {:d} from Y, margin gain: {:.5f}'.format(u, b))

    if verbose:
        print('run time: {:.5f} sec.'.format(time.time() - start_time))
        print('num of nodes: {:d}.'.format(len(X)))

    # print(sorted(list(X)))
    # print(sorted(list(Y)))

    return X


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


def test_ba_dense_graph():
    num_nodes = 1000
    first_deg = 3
    second_deg = 10
    subgraph_size = 100
    mu = 3
    restart = 0.3

    np.set_printoptions(precision=6, linewidth=2000)

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic'
    dataset = []
    for case_id in range(10):
        fn = 'dual_mu_{}_nodes_{}_deg1_{}_deg2_{}_subsize_{}_restart_{}_{}.pkl'.format(mu, num_nodes, first_deg, second_deg, subgraph_size, restart, case_id)
        with open(os.path.join(path, fn), 'rb') as rfile:
            instance = pickle.load(rfile)
            dataset.append(instance)
            print(fn)


    times = 2
    normalize = False
    print('normalize={}'.format(normalize))
    # for lmbd in range(1, 5001, 300): # normalize=True
    for lmbd in range(1, 601, 30): # normalize=False
        print('lambda {}'.format(lmbd))
        results = []
        results_2 = []
        for i, instance in enumerate(dataset):
            for time in range(times):
                print('instance {}, time {}'.format(i, time))
                first_graph, second_graph, weight, true_subgraph = instance['first_graph'], instance['second_graph'], instance['weight'], instance['true_subgraph']

                X = BFNS(first_graph, weight, lmbd, normalize=normalize, sort=False, verbose=False)

                prec, rec, fm, iou = evaluate(X, true_subgraph)

                graph_density = nx.density(first_graph)
                subgraph_density = nx.density(nx.subgraph(first_graph, X))
                num_nodes = nx.subgraph(first_graph, X).number_of_nodes()
                num_edges = nx.subgraph(first_graph, X).number_of_edges()

                print('first graph, prec={:.5f}, rec={:.5f}, fm={:.5f}, iou={:.5f}, graph density={:.5f}, subgraph density={:.5f}'.format(prec, rec, fm, iou, graph_density, subgraph_density, num_nodes, num_edges))

                results.append((prec, rec, fm, iou, graph_density, subgraph_density, num_nodes, num_edges))

                X_2 = BFNS(second_graph, weight, lmbd, normalize=normalize, sort=False, verbose=False)

                prec_2, rec_2, fm_2, iou_2 = evaluate(X_2, true_subgraph)

                graph_density_2 = nx.density(second_graph)
                subgraph_density_2 = nx.density(nx.subgraph(second_graph, X_2))
                num_nodes_2 = nx.subgraph(second_graph, X_2).number_of_nodes()
                num_edges_2 = nx.subgraph(second_graph, X_2).number_of_edges()

                print('second graph, prec={:.5f}, rec={:.5f}, fm={:.5f}, iou={:.5f}, graph density={:.5f}, subgraph density={:.5f}'.format(prec_2, rec_2, fm_2, iou_2, graph_density_2, subgraph_density_2, num_nodes_2, num_edges_2))

                results_2.append((prec_2, rec_2, fm_2, iou_2, graph_density_2, subgraph_density_2, num_nodes_2, num_edges_2))

        results = np.array(results)
        print(results)
        print('mean', np.array2string(np.mean(results, axis=0), threshold=np.inf, max_line_width=np.inf, separator=',').replace('\n', ''))
        print('std', np.array2string(np.std(results, axis=0), threshold=np.inf, max_line_width=np.inf, separator=',').replace('\n', ''))

        results_2 = np.array(results_2)
        print(results_2)
        print('mean_2', np.array2string(np.mean(results_2, axis=0), threshold=np.inf, max_line_width=np.inf, separator=',').replace('\n', ''))
        print('std_2', np.array2string(np.std(results_2, axis=0), threshold=np.inf, max_line_width=np.inf, separator=',').replace('\n', ''))





if __name__ == '__main__':

    test_ba_dense_graph()