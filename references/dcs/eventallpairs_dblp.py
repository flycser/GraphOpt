#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : eventallpairs_dblp
# @Date     : 08/02/2019 12:56:42
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


from __future__ import print_function

import os
import sys
import time
import pickle
import collections

from random import shuffle

import numpy as np
import networkx as nx

from multiprocessing import Pool, Process, Manager


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


def BFNS(graph, weight, lmbd, shortest_path_lengths=None, normalize=True, sort=True, verbose=True):
    # normalize weight
    if normalize:
        weight = (weight - np.min(weight)) / (np.max(weight) - np.min(weight))
    else:
        weight = weight - np.min(weight)

    if sort:
        weight[::-1].sort()

    nodes = [node for node in graph.nodes()]
    shuffle(nodes)

    X = set() # start from empty set
    Y = set([node for node in graph.nodes()]) # start from entire node set

    start_time = time.time()

    for i in range(graph.number_of_nodes()):
        # how to select next node? randomly? shuffle node list firstly
        u = nodes[i]
        # shortest_lengths_from_u = nx.shortest_path_length(graph, source=u)

        #   Q_AP(X\cup u) - Q_AP(X)
        # = I(X\cup u) + D_AP(V) - D_AP(X\cup u) - I(X) - D_AP(V) + D_AP(X)
        # = I(X) + I(u) + D_AP(V) - D_AP(X\cup u) - I(X) - D_AP(V) + D_AP(X)
        # = I(u) + D(X) - D_AP(X\cup u)
        Dap_a = 0
        for node in X:
            if node not in shortest_path_lengths[u].keys(): # note, handle not connected nodes
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
            if node not in shortest_path_lengths[u].keys(): # note, handle not connected nodes
                Dap_b += max(shortest_path_lengths[u].values())
            else:
                Dap_b += shortest_path_lengths[u][node]
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

    print(sorted(list(X)))
    print(sorted(list(Y)))

    return X


def Max_Cut():
    pass


def generate_dataset(verbose=True):
    num_nodes = 1000
    deg = 5
    subgraph_size = 100

    # generate graph with barabasi albert model
    graph = nx.barabasi_albert_graph(num_nodes, deg)


    # get degrees of all nodes
    num_nodes = graph.number_of_nodes()
    node_degree = np.zeros(num_nodes)
    for current_node in graph.nodes():
        node_degree[current_node] = nx.degree(graph, current_node)

    # random walk, biased for nodes with higher degrees
    restart = 0.3
    start_node = next_node = np.random.choice(range(num_nodes))
    subgraph = set()
    subgraph.add(start_node)
    while True:
        if len(subgraph) >= subgraph_size:
            break

        neighbors = [node for node in nx.neighbors(graph, next_node)]

        neighbor_degree_dist = [node_degree[node] for node in neighbors]
        sum_prob = np.sum(neighbor_degree_dist)
        # note, when there are no neighbors for one node on conceptual network, its probabilities to other neighbor nodes are equal
        normalized_prob_dist = [prob / sum_prob if not sum_prob == 0. else 1. / len(neighbors) for prob in
                                neighbor_degree_dist]

        if np.random.uniform() > restart:
            next_node = np.random.choice(neighbors, p=normalized_prob_dist)  # biased for those nodes with high degree
        else:  # restart
            next_node = start_node

        subgraph.add(next_node)
        if verbose:
            print('generating {:d} nodes ...'.format(len(subgraph)))

    mean_1 = 5.
    mean_2 = 0.
    std = 1.
    weight = np.zeros(graph.number_of_nodes())
    for node in graph.nodes():
        if node in subgraph:
            weight[node] = np.random.normal(mean_1, std)
        else:
            weight[node] = np.random.normal(mean_2, std)

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic'
    fn = 'nodes_{}_deg_{}_subsize_{}.pkl'.format(num_nodes, deg, subgraph_size)

    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump({'graph': graph, 'true_subgraph': subgraph, 'weight': weight}, wfile)

    if verbose:
        print('num of nodes: {:d}'.format(graph.number_of_nodes()))
        print('num of edges: {:d}'.format(graph.number_of_edges()))

        density = nx.density(graph)
        print('graph density', density)

        subgraph = nx.subgraph(graph, subgraph)
        density = nx.density(subgraph)
        print('subgraph density', density)

        # degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)  # degree sequence
        # degreeCount = collections.Counter(degree_sequence)
        # deg, cnt = zip(*degreeCount.items())
        # print(deg)
        # print(cnt)


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

def dense_projection(graph, weight, threshold, progression, normalize=False, sort=False, times=2):
    density_subgraph_list = []

    print('dense projection')

    manager = Manager()
    return_dict = manager.dict()
    jobs = []

    for lmbd in progression:
        verbose = False

        para = graph, weight, lmbd, normalize, sort, verbose

        process = Process(target=BFNS_worker, args=(para, return_dict))
        jobs.append(process)
        process.start()

    for proc in jobs:
        proc.join()

    for b in return_dict.keys():
        return_dict[b]

def BFNS_worker(para, return_dict):
    graph, weight, lmbd, normalize, sort, verbose = para

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
            if node not in shortest_lengths_from_u.keys():  # note, handle not connected nodes
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
            if node not in shortest_lengths_from_u.keys():  # note, handle not connected nodes
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
            # if verbose:
            #     print('add node {:d} to X, margin gain: {:.5f}'.format(u, a))
        else:
            Y.remove(u)
            # if verbose:
            #     print('remove node {:d} from Y, margin gain: {:.5f}'.format(u, b))

    if verbose:
        print('{} run time: {:.5f} sec.'.format(lmbd, time.time() - start_time))

    return_dict[lmbd] = X


if __name__ == '__main__':

    # generate synthetic dataset
    # generate_dataset()

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/DBLP/DBLP_Citation_2014_May/DM'
    fn = 'dm_top30000_dataset.pkl'

    with open(os.path.join(path, fn), 'rb') as rfile:
        dataset = pickle.load(rfile)

    second_graph = dataset['second_graph']

    print(second_graph.number_of_nodes())
    print(second_graph.number_of_edges())

    lcc = max(nx.connected_component_subgraphs(second_graph), key=len)
    subgraph = nx.subgraph(second_graph, lcc)
    print(nx.is_connected(subgraph))
    print(subgraph.number_of_nodes())
    print(subgraph.number_of_edges())

    first_graph = dataset['first_graph']
    subgraph = nx.subgraph(first_graph, lcc)
    print(nx.is_connected(subgraph))
    print(subgraph.number_of_nodes())
    print(subgraph.number_of_edges())


    # start_time = time.time()
    # shortest_path_lengths = {}
    # for node in second_graph.nodes():
    #     x = nx.shortest_path_length(second_graph, source=node)
    #     shortest_path_lengths[node] = x
    # print('shortest_path time: {:.5f}'.format(time.time() - start_time))

    # print(num_nodes)
    # print(graph.number_of_edges())
    # for edge in graph.edges():
    #     print(edge[0], edge[1])

    # # true_subgraph = dataset['true_subgraph']
    # weight = dataset['weight'] #
    # weight_normalize = sys.argv[2].lower() == 'true'
    # if weight_normalize:
    #     weight = dataset[weight] / float(max(dataset[weight]))
    #
    # # note, weight should be normalized
    # lmbd = int(sys.argv[1])
    # # lmbd = 100
    # # subgraph = greedAP(graph, weight, lmbd, normalize=False, sort=True)
    # subgraph = BFNS(second_graph, weight, lmbd, normalize=True, sort=False, shortest_path_lengths=shortest_path_lengths)

    density = nx.density(second_graph)
    print('density of graph: {:.5f}'.format(density))

    print(sorted(list(subgraph)))
    print(len(subgraph))

    # density = nx.density(nx.subgraph(graph, true_subgraph))
    # print('density of true subgraph: {:.5f}'.format(density))

    original_subgraph = [86, 96, 247, 399, 400, 508, 513, 578,579,  591,602,672,675,753,
 90,789,801,888,901,925, 992,1078,1135,1146,1260,1406 ,1423, 1440,1574,
 91,1632 ,1636 ,1769, 1822 ,1836 ,1908, 1912 ,1951, 1969, 2040, 2072 ,2179 ,2180 ,2202,
 92,2377, 2392 ,2398 ,2408, 2442, 2447, 2626 ,2711, 2799, 2839 ,2871 ,2910 ,2960 ,2992,
 93 , 3036 ,3162 ,3194, 3228, 3246 ,3259, 3308 ,3385, 3393, 3485 ,3614 ,3624 ,3650, 3764,
 94 , 3771 ,3843, 3850, 3991,3996,4094]

    subgraph = nx.subgraph(first_graph, original_subgraph)
    print(subgraph.number_of_nodes())
    print(subgraph.number_of_edges())
    print(nx.is_connected(subgraph))
    lcc = max(nx.connected_component_subgraphs(subgraph), key=len)
    print(len(lcc))


    # subgraph = nx.subgraph(second_graph, original_subgraph)
    subgraph = nx.subgraph(second_graph, lcc)
    print(subgraph.number_of_nodes())
    print(subgraph.number_of_edges())

    print(nx.is_connected(subgraph))
    lcc = max(nx.connected_component_subgraphs(subgraph), key=len)
    print(len(lcc))

    subgraph = nx.subgraph(second_graph, lcc)

    density = nx.density(nx.subgraph(second_graph, subgraph))
    print('density of subgraph: {}'.format(density))

    fn = 'dm_author_map.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        dm_author_map = pickle.load(rfile)

    new_old = dm_author_map['new_old']
    old_new = dm_author_map['old_new']

    # print(dm_author_map.keys())

    fn = 'dm_authors.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        dm_author = pickle.load(rfile)

    for node in subgraph.nodes():
        for name in dm_author.keys():
            if dm_author[name] == new_old[node]:
                print(node, new_old[node], name)

    for edge in subgraph.edges():
        node_1, node_2 = edge
        # print(node_1, node_2, new_old[node_1], new_old[node_2])
        for name in dm_author.keys():
            if dm_author[name] == new_old[node_1]:
                name_1 = name

            if dm_author[name] == new_old[node_2]:
                name_2 = name

        print(node_1, node_2, new_old[node_1], new_old[node_2], name_1, name_2)

    # prec, rec, fm, iou = evaluate(subgraph, true_subgraph)
    # print('precision: {:.5f}, recall: {:.5f}, f-measure: {:.5f}, iou: {:.5f}'.format(prec, rec, fm, iou))