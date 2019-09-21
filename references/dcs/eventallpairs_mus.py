#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : eventallpairs_mus
# @Date     : 08/05/2019 17:21:11
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


from __future__ import print_function

import os
import time
import pickle
import collections

from random import shuffle

import numpy as np
import networkx as nx


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


def BFNS(graph, weight, lmbd, start_node, lcc_diameter, normalize=True, sort=True, verbose=True):
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

    if start_node > 0:
        X.add(start_node)
        Y.remove(start_node)

    start_time = time.time()

    for i in range(graph.number_of_nodes()):
        # how to select next node? randomly? shuffle node list firstly
        u = nodes[i]
        if u == start_node:
            continue

        shortest_lengths_from_u = nx.shortest_path_length(graph, source=u)

        #   Q_AP(X\cup u) - Q_AP(X)
        # = I(X\cup u) + D_AP(V) - D_AP(X\cup u) - I(X) - D_AP(V) + D_AP(X)
        # = I(X) + I(u) + D_AP(V) - D_AP(X\cup u) - I(X) - D_AP(V) + D_AP(X)
        # = I(u) + D(X) - D_AP(X\cup u)
        Dap_a = 0
        for node in X:
            if node not in shortest_lengths_from_u.keys(): # note, handle not connected nodes
                # Dap_a -= max(shortest_lengths_from_u.values())
                Dap_a -= (max(shortest_lengths_from_u.values()) + lcc_diameter)
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
                # Dap_b += max(shortest_lengths_from_u.values())
                Dap_b += (max(shortest_lengths_from_u.values()) + lcc_diameter)
            else:
                Dap_b += shortest_lengths_from_u[node]

        b = - lmbd * weight[u] + Dap_b

        a = a if a > 0 else 0
        b = b if b > 0 else 0

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


if __name__ == '__main__':

    # generate synthetic dataset
    # generate_dataset()

    case_id = 0
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Mus_Multiplex_Genetic/dual'
    fn = 'dual_mus_case_{}.pkl'.format(case_id)

    with open(os.path.join(path, fn), 'rb') as rfile:
        dataset = pickle.load(rfile)

    graph = dataset['second_graph']

    # print(num_nodes)
    # print(graph.number_of_edges())
    # for edge in graph.edges():
    #     print(edge[0], edge[1])

    true_subgraph = dataset['true_subgraph']

    weight = dataset['weight']
    lcc = max(nx.connected_component_subgraphs(graph), key=len)
    tmp = 0
    start_node = 0
    for node in lcc:
        if weight[node] > tmp:
            tmp = weight[node]
            start_node = node

    print(start_node, weight[start_node])
    # note, weight should be normalized
    lmbd = 50
    # subgraph = greedAP(graph, weight, lmbd, normalize=False, sort=True)
    # subgraph = BFNS(graph, weight, lmbd, start_node, lcc_diameter=9, normalize=True, sort=True)
    subgraph = BFNS(graph, weight, lmbd, start_node, lcc_diameter=19, normalize=False, sort=True)

    density = nx.density(graph)
    print('density of graph: {:.5f}'.format(density))

    print(sorted(list(true_subgraph)))
    print(sorted(list(subgraph)))
    print(len(subgraph))

    density = nx.density(nx.subgraph(graph, true_subgraph))
    print('density of true subgraph: {:.5f}'.format(density))

    print(sorted([node for node in nx.isolates(graph)]))
    print(sorted(list(set([node for node in nx.isolates(graph)])&set(list(true_subgraph)))))
    # lcc = max(nx.connected_component_subgraphs(graph), key=len)
    # print(nx.subgraph(graph, lcc).number_of_nodes())
    # print(nx.diameter(nx.subgraph(graph, lcc)))

    density = nx.density(nx.subgraph(graph, subgraph))
    print(nx.subgraph(graph, subgraph).number_of_nodes())
    print(nx.subgraph(graph, subgraph).number_of_edges())
    # print('density of subgraph: {:.5f}'.format(density))
    print('density of subgraph: {}'.format(density))

    prec, rec, fm, iou = evaluate(subgraph, true_subgraph)
    print('precision: {:.5f}, recall: {:.5f}, f-measure: {:.5f}, iou: {:.5f}'.format(prec, rec, fm, iou))