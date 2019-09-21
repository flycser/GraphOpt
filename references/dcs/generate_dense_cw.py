#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : generate_dense_cw
# @Date     : 08/11/2019 14:50:14
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :

from __future__ import print_function

import pickle

import numpy as np
import networkx as nx


def generate_graph(n, k, mu):

    graph = nx.complete_graph(k)
    for node in range(k, n):
        graph.add_node(node)

    # print(graph.nodes())
    # print(graph.edges())

    for i in range(k, n):
        for j in range(2):
            target = np.random.randint(n)
            while target == i or nx.degree(graph, target) == k - 2:
                target = np.random.randint(n)

            graph.add_edge(i, target)

    # print(graph.nodes())
    # print(graph.edges())
    print(graph.number_of_nodes())
    print(graph.number_of_edges())
    print(nx.is_connected(graph))
    print(nx.density(graph))


    attributes = np.zeros(n)
    for node in range(n):
        if node < k:
            # attributes[node] = np.random.normal(mu, 1)
            attributes[node] = 1.
        else:
            # attributes[node] = np.random.normal(0, 1)
            attributes[node] = 0.

    # add 20% noise
    x = []
    for node in range(n):
        if np.random.uniform(0., 1.) < 0.2:
            print(node, end=',')
            x.append(node)
            attributes[node] = np.random.uniform(0., 1.)
    print()
    print(len(x))

    return graph, attributes

    # fn = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Test_EventAllPairs/instance_mu5_.pkl'
    # with open(fn, 'wb') as wfile:
    #     pickle.dump({'graph='})


    # for edge in graph.edges():
    #     print(edge[0], edge[1])


if __name__ == '__main__':

    pass

    n = 1000
    k = 100
    mu = 3

    dataset = []
    for case_id in range(10):
        graph, attributes = generate_graph(n, k, mu)

        dataset.append({'graph': graph, 'attributes': attributes})

    # fn = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Test_EventAllPairs/mu{}.pkl'.format(mu)
    fn = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Test_EventAllPairs/0_1_noise_20.pkl'
    with open(fn, 'wb') as wfile:
        pickle.dump(dataset, wfile)