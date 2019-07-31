#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : shortest_path
# @Date     : 07/29/2019 18:04:20
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     : shortest path api test


from __future__ import print_function

import os
import time
import pickle
import networkx as nx


if __name__ == '__main__':


    # graph = nx.Graph()
    # graph.add_edge(0, 1)
    # graph.add_edge(0, 3)
    # graph.add_edge(0, 4)
    # graph.add_edge(1, 2)
    # graph.add_edge(2, 3)
    # graph.add_edge(2, 4)
    # graph.add_edge(3, 4)

    # x = nx.shortest_path_length(graph)
    #
    # print(list(x))


    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/app2/ba'
    fn = 'train_100000_deg_3.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        dataset = pickle.load(rfile)

    graph = dataset[0]['graph']

    start = time.time()
    # x = nx.shortest_path_length(graph)
    x = nx.shortest_path_length(graph, target=0)
    end = time.time()
    # print(type(x))
    print(end - start)
    #
    # for a in x:
    #     print(a)

    print(x)

    graph = nx.Graph()
    graph.add_edge(0, 1)
    graph.add_edge(0, 3)
    graph.add_edge(0, 4)
    graph.add_edge(1, 2)
    graph.add_edge(1, 5)
    graph.add_edge(2, 3)
    graph.add_edge(2, 4)
    graph.add_edge(2, 6)
    graph.add_edge(3, 4)
    graph.add_edge(5, 6)
    graph.add_edge(5, 7)

    x = nx.density(graph)
    print(x)

    subset = [0, 3, 4, 1, 2]
    subgraph = nx.subgraph(graph, subset)

    y = nx.density(subgraph)
    print(y)