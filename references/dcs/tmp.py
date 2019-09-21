#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : tmp
# @Date     : 09/04/2019 23:53:22
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :

from __future__ import print_function

import numpy as np
import networkx as nx


def adj_matrix(graph):
    A = np.zeros((graph.number_of_nodes(), graph.number_of_nodes()))
    for node_1, node_2 in graph.edges():
        A[node_1][node_2] = 1.
        A[node_2][node_1] = 1.

    return A


if __name__ == '__main__':

    graph = nx.Graph()
    graph.add_edge(0, 3)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(3, 5)

    edges = np.array(graph.edges)

    print(edges)

    A = adj_matrix(graph)
    print(A)
