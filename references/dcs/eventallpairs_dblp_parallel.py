#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : eventallpairs_dblp_parallel
# @Date     : 08/02/2019 20:52:50
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

def dense_projection(graph, weight, threshold, progression, normalize=False, sort=False):
    start_time = time.time()
    shortest_path_lengths = {}
    for node in second_graph.nodes():
        x = nx.shortest_path_length(second_graph, source=node)
        shortest_path_lengths[node] = x
    print('shortest_path time: {:.5f}'.format(time.time() - start_time))

    density_subgraph_list = []

    print('dense projection')

    manager = Manager()
    return_dict = manager.dict()
    jobs = []

    for lmbd in progression:
        verbose = True

        para = graph, weight, lmbd, shortest_path_lengths, normalize, sort, verbose

        process = Process(target=BFNS_worker, args=(para, return_dict))
        jobs.append(process)
        process.start()

    for proc in jobs:
        proc.join()

    for b in return_dict.keys():
        x = return_dict[b]
        print(x)
        subgraph = nx.subgraph(graph, x)
        print(nx.density(subgraph))



def BFNS_worker(para, return_dict):
    graph, weight, lmbd, shortest_path_lengths, normalize, sort, verbose = para

    # normalize weight
    if normalize:
        weight = (weight - np.min(weight)) / (np.max(weight) - np.min(weight))
    else:
        weight = weight - np.min(weight)

    if sort:
        weight[::-1].sort()

    # shuffle

    nodes = [node for node in graph.nodes()]
    shuffle(nodes)

    X = set()  # start from empty set
    Y = set([node for node in graph.nodes()])  # start from entire node set

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
    weight = dataset['weight'] #
    weight_normalize = sys.argv[1].lower() == 'true'
    if weight_normalize:
        weight = weight / float(max(weight))

    # # note, weight should be normalized
    # lmbd = int(sys.argv[1])
    # # lmbd = 100
    # # subgraph = greedAP(graph, weight, lmbd, normalize=False, sort=True)
    # subgraph = BFNS(second_graph, weight, lmbd, normalize=True, sort=False, shortest_path_lengths=shortest_path_lengths)


    start = 5
    ratio = 10
    progression = [start * ratio ** i for i in range(1, 6)]
    threshold = 0.003
    dense_projection(second_graph, weight, threshold, progression, normalize=True, sort=False)

    density = nx.density(second_graph)
    print('density of graph: {:.5f}'.format(density))

    print(sorted(list(subgraph)))
    print(len(subgraph))

    # density = nx.density(nx.subgraph(graph, true_subgraph))
    # print('density of true subgraph: {:.5f}'.format(density))

    # subgraph = [31, 66, 69, 71, 101, 120, 143, 172, 178]
    #
    # subgraph = nx.subgraph(second_graph, subgraph)
    # print(subgraph.number_of_nodes())
    # print(subgraph.number_of_edges())
    #
    # print(nx.is_connected(subgraph))
    # lcc = max(nx.connected_component_subgraphs(subgraph), key=len)
    # print(len(lcc))
    #
    # subgraph = nx.subgraph(second_graph, lcc)
    #
    # density = nx.density(nx.subgraph(second_graph, subgraph))
    # print('density of subgraph: {}'.format(density))
    #
    # fn = 'dm_author_map.pkl'
    # with open(os.path.join(path, fn), 'rb') as rfile:
    #     dm_author_map = pickle.load(rfile)
    #
    # new_old = dm_author_map['new_old']
    # old_new = dm_author_map['old_new']
    #
    # # print(dm_author_map.keys())
    #
    # fn = 'dm_authors.pkl'
    # with open(os.path.join(path, fn), 'rb') as rfile:
    #     dm_author = pickle.load(rfile)
    #
    # for node in subgraph.nodes():
    #     for name in dm_author.keys():
    #         if dm_author[name] == new_old[node]:
    #             print(node, new_old[node], name)
    #
    # for edge in subgraph.edges():
    #     node_1, node_2 = edge
    #     # print(node_1, node_2, new_old[node_1], new_old[node_2])
    #     for name in dm_author.keys():
    #         if dm_author[name] == new_old[node_1]:
    #             name_1 = name
    #
    #         if dm_author[name] == new_old[node_2]:
    #             name_2 = name
    #
    #     print(node_1, node_2, new_old[node_1], new_old[node_2], name_1, name_2)
