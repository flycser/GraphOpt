#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : eventallpairs_test_2
# @Date     : 08/19/2019 01:47:52
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


def greedAP(graph, weight, lmbd, normalize=True, sort=False, verbose=True):

    # normalize weight
    if normalize:
        weight = (weight - np.min(weight)) / (np.max(weight) - np.min(weight))
    else:
        weight = weight - np.min(weight) # note, positive

    print(weight)

    if sort:
        weight[::-1].sort()

    print(weight)

    nodes = [node for node in graph.nodes()]
    shuffle(nodes)

    X = set() # start from empty set

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


def test(mu):

    # fn = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Test_EventAllPairs/mu{}.pkl'.format(mu)
    fn = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Test_EventAllPairs/0_1_noise_20.pkl'
    with open(fn, 'rb') as rfile:
        dataset = pickle.load(rfile)

    # lmbd = 100
    times = 2
    normalize = False
    print(mu, normalize)
    for lmbd in range(501, 1001, 10):
        print('lambda {}'.format(lmbd))
        # run all instances, 2 times for each instance
        results = []
        for i, instance in enumerate(dataset):
            for time in range(times):
                print('instance {}, time {}'.format(i, time))
                graph, weight = instance['graph'], instance['attributes']

                X = BFNS(graph, weight, lmbd, normalize=normalize, sort=False, verbose=False)

                true_subgraph = range(100)

                prec, rec, fm, iou = evaluate(X, true_subgraph)

                print('prec={:.5f}, rec={:.5f}, fm={:.5f}, iou={:.5f}'.format(prec, rec, fm, iou))

                results.append((prec, rec, fm, iou))

            # if i == 1:
            #     break

        results = np.array(results)
        print(results)
        print('mean', np.mean(results, axis=0))
        print('std', np.std(results, axis=0))


def test_ba_dense_graph():
    elevate_val = 1

    fn = '/network/rit/lab/ceashpc/share_data/GraphOpt/DCS/dense/syn/dataset_ele{}.pkl'.format(elevate_val)

    with open(fn, 'rb') as rfile:
        dataset = pickle.load(rfile)

    # lmbd = 100
    times = 2
    normalize = True
    # normalize = False
    print('normalize={}, elevate={}'.format(normalize, elevate_val))
    # for lmbd in [2500]:
    for lmbd in range(500, 5001, 500):
    # for lmbd in range(1011, 1502, 10):
        print('lambda {}'.format(lmbd))
        # run all instances, 2 times for each instance
        results = []
        for i, instance in enumerate(dataset):
            for time in range(times):
                print('instance {}, time {}'.format(i, time))
                graph, weight, true_subgraph = instance['graph'], instance['weight'], instance['true_subgraph']

                X = BFNS(graph, weight, lmbd, normalize=normalize, sort=False, verbose=False)

                prec, rec, fm, iou = evaluate(X, true_subgraph)

                graph_density = nx.density(graph)
                subgraph_density = nx.density(nx.subgraph(graph, X))
                num_nodes = nx.subgraph(graph, X).number_of_nodes()
                num_edges = nx.subgraph(graph, X).number_of_edges()

                print('prec={:.5f}, rec={:.5f}, fm={:.5f}, iou={:.5f}, graph density={:.5f}, subgraph density={:.5f}'.format(prec, rec, fm, iou, graph_density, subgraph_density, num_nodes, num_edges))

                results.append((prec, rec, fm, iou, graph_density, subgraph_density, num_nodes, num_edges))

            # if i == 1:
            #     break

        results = np.array(results)
        print(results)
        print('mean', np.mean(results, axis=0))
        print('std', np.std(results, axis=0))


if __name__ == '__main__':

    # mu = int(sys.argv[1])
    # test(mu)
    # test(None)

    # test_ba_dense_graph()

    print('test')