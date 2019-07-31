#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : generate_dual_networks
# @Date     : 07/31/2019 10:06:57
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     : generate synthetic dual networks


from __future__ import print_function

import os
import sys
import pickle

import numpy as np
import networkx as nx

def generate_instance(verbose=True):
    num_nodes = 1000
    first_deg = 3
    second_deg = 10
    subgraph_size = 100

    # generate graph with barabasi albert model
    first_graph = nx.barabasi_albert_graph(num_nodes, first_deg)
    second_graph = nx.barabasi_albert_graph(num_nodes, second_deg)

    # get degrees of all nodes in the second graph
    second_graph_node_degree = np.zeros(num_nodes)
    for current_node in second_graph.nodes():
        second_graph_node_degree[current_node] = nx.degree(second_graph, current_node)

    # random walk on the first graph, biased for nodes with higher degrees in second graph
    restart = 0.3
    start_node = next_node = np.random.choice(range(num_nodes))
    subgraph = set()
    subgraph.add(start_node)
    while True:
        if len(subgraph) >= subgraph_size:
            break

        neighbors = [node for node in nx.neighbors(first_graph, next_node)] # neighbors of the next node in the first graph

        neighbor_degree_dist = [second_graph_node_degree[node] for node in neighbors] # visit those nodes with higher degrees in the second graph

        sum_prob = np.sum(neighbor_degree_dist)
        # note, when there are no neighbors for one node on conceptual network, its probabilities to other neighbor nodes are equal
        normalized_prob_dist = [prob / sum_prob if not sum_prob == 0. else 1. / len(neighbors) for prob in neighbor_degree_dist]

        if np.random.uniform() > restart: # random walk to next node
            next_node = np.random.choice(neighbors,
                                         p=normalized_prob_dist)  # biased for those nodes with high degree
        else:  # restart from initial node
            next_node = start_node

        subgraph.add(next_node)
        if verbose:
            print('generating {:d} nodes ...'.format(len(subgraph)))

    # generate node weight
    mean_1 = 5.
    mean_2 = 0.
    std = 1.
    weight = np.zeros(num_nodes)
    for node in first_graph.nodes():
        if node in subgraph:
            weight[node] = np.random.normal(mean_1, std)
        else:
            weight[node] = np.random.normal(mean_2, std)

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic'
    fn = 'dual_nodes_{}_deg1_{}_deg2_{}_subsize_{}.pkl'.format(num_nodes, first_deg, second_deg, subgraph_size)

    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump({'first_graph': first_graph, 'second_graph': second_graph, 'true_subgraph': subgraph, 'weight': weight}, wfile)

    if verbose:
        print('num of nodes: {:d}'.format(num_nodes))
        print('num of edges: {:d} in first graph'.format(first_graph.number_of_edges()))
        print('num of edges: {:d} in second graph'.format(second_graph.number_of_edges()))

        density = nx.density(first_graph)
        print('first graph density', density)

        density = nx.density(second_graph)
        print('second graph density', density)

        first_subgraph = nx.subgraph(first_graph, subgraph)
        density = nx.density(first_subgraph)
        print('subgraph density in first graph', density)

        second_subgraph = nx.subgraph(second_graph, subgraph)
        density = nx.density(second_subgraph)
        print('subgraph density in second graph', density)


if __name__ == '__main__':

    generate_instance()