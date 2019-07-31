#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : ba_dataset
# @Date     : 07/17/2019 22:16:10
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


from __future__ import print_function

import os
import pickle
import collections

import numpy as np
import networkx as nx

def convert():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/app2/ba'
    train_dataset = []

    nodes = 100000
    block = 100
    mu = 5
    subsize = 5000.0
    deg = 10
    for id in range(2):
        fn = 'nodes_{}_blocks_{}_mu_{}_subsize_{}_{}_deg_{}_train_{}.pkl'.format(nodes, block, mu, subsize, subsize, deg, id)
        with open(os.path.join(path, fn), 'rb') as rfile:
            # instance = pickle.load(rfile)
            instance = pickle.load(rfile)[0]
            # print(type(instance['features']))
            instance['features'] = np.array(instance['features'])
            train_dataset.append(instance)


    # train_dataset.append(instance[0])
    # print(instance[0].keys())

    fn = 'train_{}_deg_{}.pkl'.format(nodes, deg)
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(train_dataset, wfile)

def rename():

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/log/ghtp/block_ba'
    for fn in os.listdir(path):
        src = fn
        dst = os.path.splitext(fn)[0] + '_serial' + '.txt'
        os.rename(os.path.join(path, src), os.path.join(path, dst))
        # os.rename(src, dst)
        # print(dst)

def generate_dcs_amen_dataset(id):

    num_nodes = 10000
    deg = 3

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/app2/ba'
    fn = 'train_{}_deg_{}.pkl'.format(num_nodes, deg)
    with open(os.path.join(path, fn), 'rb') as rfile:
        dataset = pickle.load(rfile)

    graph = dataset[id]['graph']

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/DCS/amen'
    fn = 'biased_{}_instance_{}.pkl'.format(num_nodes, id)
    with open(os.path.join(path, fn), 'rb') as rfile:
        dataset = pickle.load(rfile)

    # features = dataset[id]['features']
    # true_subgraph = dataset[id]['true_subgraph']
    # graph = dataset[id]['graph']

    features = dataset['attributes']
    true_subgraph = dataset['subgraph']

    print(nx.density(nx.subgraph(graph, true_subgraph)))

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/DCS/amen'

    fn = 'instance_{}.txt'.format(id)
    with open(os.path.join(path, fn), 'w') as wfile:
        wfile.write('{} {}\n'.format(graph.number_of_nodes(), 1))
        for node in graph.nodes():
            wfile.write('{}\n'.format(features[node]))

        wfile.write('{}\n'.format(graph.number_of_edges()))
        for node_1, node_2 in graph.edges():
            node_1, node_2 = (node_1, node_2) if node_1 < node_2 else (node_2, node_1)
            wfile.write('{} {} 1.0\n'.format(node_1, node_2))

        wfile.write(' '.join([str(node) for node in sorted(list(true_subgraph))]) + '\n')
        wfile.write('0')


def generate_ba_dcs_biased(id):
    num_nodes = 10000
    deg = 3
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/app2/ba'
    fn = 'train_{}_deg_{}.pkl'.format(num_nodes, deg)
    with open(os.path.join(path, fn), 'rb') as rfile:
        dataset = pickle.load(rfile)

    features = dataset[id]['features']
    true_subgraph = dataset[id]['true_subgraph']
    graph = dataset[id]['graph']

    num_nodes = graph.number_of_nodes()
    node_degree = np.zeros(num_nodes)
    for current_node in graph.nodes():
        node_degree[current_node] = nx.degree(graph, current_node)

    start_node = next_node = np.random.choice(range(num_nodes))
    subgraph = set()
    subgraph.add(start_node)
    restart = 0.1
    count = 1000
    print(start_node)
    # random walk on undirected physical graph, biased for nodes with higher degree on conceptual network
    while True:
        if len(subgraph) >= count:
            break

        neighbors = [node for node in nx.neighbors(graph, next_node)]

        neighbor_degree_dist = [node_degree[node] for node in neighbors]
        sum_prob = np.sum(neighbor_degree_dist)
        # note, when there are no neighbors for one node on conceptual network, its probabilities to other neighbor nodes are equal
        normalized_prob_dist = [prob / sum_prob if not sum_prob == 0. else 1. / len(neighbors) for prob in neighbor_degree_dist]
        # if sum_prob == 0.:
        #     print(len(neighbors))
        #     print(neighbor_degree_dist)
        #     print(sum_prob)
        #     print(normalized_prob_dist)
        #     break
        # print(len(neighbor_degree_dist))
        # print(normalized_prob_dist)

        if np.random.uniform() > restart:
            next_node = np.random.choice(neighbors, p=normalized_prob_dist)  # biased for those nodes with high degree
        else:  # restart
            next_node = start_node

        subgraph.add(next_node)
        print(len(subgraph))

    mean_1 = 5.
    mean_2 = 0.
    std = 1.
    attributes = np.zeros(graph.number_of_nodes())
    for node in graph.nodes():
        if node in subgraph:
            attributes[node] = np.random.normal(mean_1, std)
        else:
            attributes[node] = np.random.normal(mean_2, std)

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/DCS/amen'
    fn = 'biased_{}_instance_{}.pkl'.format(num_nodes, id)
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump({'attributes': attributes, 'subgraph': subgraph}, wfile)

    subgraph = nx.subgraph(graph, subgraph)
    density = nx.density(subgraph)
    print('density', density)

    true_subgraph = nx.subgraph(graph, true_subgraph)
    density = nx.density(true_subgraph)
    print('density', density)

    degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    print(deg)
    print(cnt)

    print()

if __name__ == '__main__':

    generate_dcs_amen_dataset(id=0)

    # generate_ba_dcs_biased(id=0)

    # convert()

    # path = '/network/rit/lab/ceashpc/share_data/GraphOpt/app2/ba'
    # for id in range(10):
    #     fn = 'train_{}.pkl'.format(1000*(id+1))
    #     with open(os.path.join(path, fn), 'rb') as rfile:
    #         dataset = pickle.load(rfile)
    #         print(len(dataset))