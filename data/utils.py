#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : utils
# @Date     : 03/27/2019 22:39:21
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


from __future__ import print_function

import numpy as np
import networkx as nx


def subgraph_random_walk(graph, num_nodes_subgraph, restart=0.1):

    # strategy 1
    current_node = start_node = np.random.choice(list(graph.nodes))
    subgraph = set()
    subgraph.add(start_node)
    while True:
        if len(subgraph) >= num_nodes_subgraph:
            break

        neighbor_nodes = [node for node in nx.neighbors(graph, current_node)]
        next_node = np.random.choice(neighbor_nodes)
        if next_node not in subgraph:
            subgraph.add(next_node)

        current_node = next_node

    return list(subgraph)


def visual_grid_graph(width, height, nonzeros):

    print()
    print('  ', end='')
    for col_ind in range(width):
        print('{:>3d}'.format(col_ind), end='')
    print()
    print()

    for i in range(height):
        for j in range(width):
            node_ind = i * width + j
            if 0 == j:
                print('{:<2d}'.format(i), end='')

            if node_ind in nonzeros:
                print('{:>3}'.format('*'), end='')
                # print(node_ind)
            else:
                print('{:>3}'.format('-'), end='')
            if width - 1 == j:
                print()

def visual_grid_graph_feature(width, height, features):
    print()
    print('  ', end='')
    for col_ind in range(width):
        print('{:>8d}'.format(col_ind), end='')
    print()
    print()

    for i in range(height):
        for j in range(width):
            node_ind = i * width + j
            if 0 == j:
                print('{:<2d}'.format(i), end='')

            print('{:>8.5f}'.format(features[node_ind]), end='')
            if width - 1 == j:
                print()