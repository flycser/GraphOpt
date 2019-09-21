#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : generate_er_graph
# @Date     : 08/11/2019 10:32:44
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


from __future__ import print_function


import os

import networkx as nx


if __name__ == '__main__':
    # n = 1000
    # p = 0.007
    # graph = nx.erdos_renyi_graph(n, p)
    #
    # print(nx.is_connected(graph))
    # if nx.is_connected(graph):
    #     fn = 'edges_3.txt'
    #     with open(fn, 'w') as wfile:
    #         for edge in graph.edges():
    #             wfile.write('{} {}'.format(edge[0], edge[1]))
    #             print('{} {}'.format(edge[0], edge[1]))

    fn = 'edges_3.txt'
    with open(fn) as rfile:
        graph = nx.read_edgelist(rfile, delimiter=' ', nodetype=int)

    print(graph.number_of_nodes())
    print(graph.number_of_edges())


