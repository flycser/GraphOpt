#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : tmp
# @Date     : 04/03/2019 12:20:30
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


from __future__ import print_function

import os
import sys

import pickle

import numpy as np
import networkx as nx


if __name__ == '__main__':

    # num_instances = 1
    # num_nodes = 100
    # num_time_stamps = 9
    # num_time_stamps_signal = 5
    # start_time_stamps = num_time_stamps / 2 - num_time_stamps_signal / 2
    # end_time_stamps = num_time_stamps / 2 + num_time_stamps_signal / 2
    # num_nodes_subgraph_min = 10
    # num_nodes_subgraph_max = 20
    # overlap_ratio = 0.8
    # mu_0 = 0.
    # mu_1 = 5.
    #
    # path = '/network/rit/lab/ceashpc/share_data/GraphOpt/synthetic'
    # fn = 'syn_mu_{:.1f}_{:.1f}_num_{:d}_{:d}_{:d}_{:d}_time_{:d}_{:d}_{:d}.pkl'.format(mu_0, mu_1, num_instances, num_nodes, num_nodes_subgraph_min, num_nodes_subgraph_max, start_time_stamps, end_time_stamps, num_time_stamps)
    # rfn = os.path.join(path, fn)
    # with open(rfn, 'rb') as rfile:
    #     dataset = pickle.load(rfile)
    #
    # instance = dataset[0]
    # for feature in instance['features']:
    #     print(np.where(feature > 3))
    #
    # for subgraph in instance['subgraphs']:
    #     print(subgraph)


    # x = np.array([1, 2, 3, 4])
    # gradient_x = np.array([0.1, 0.2, 0.3, 0.4])
    # omega_x = [1, 2]
    # indicator_x = np.zeros(len(x))
    # indicator_x[list(omega_x)] = 1.
    #
    # x = x + 0.5 * gradient_x * indicator_x
    #
    # print(x)

    x = np.zeros(6)

    G = nx.path_graph(4)
    G.add_edge(5, 6)
    print(G.nodes)
    graphs = list(nx.connected_component_subgraphs(G))
    for graph in graphs:
        print(graph.nodes())

    x[graphs[0].nodes] = 1.
    print(x)