#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : generate_twitter_dataset
# @Date     : 08/03/2019 23:13:11
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


from __future__ import print_function

import os
import numpy as np
import networkx as nx


def generate_dataset():
    path = '/network/rit/lab/ceashpc/share_data/Twitter/data'
    fn = 'higgs-social_network.edgelist'

    first_graph = nx.Graph()
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            terms = line.strip().split(' ')
            node_1, node_2 = terms[0], terms[1]
            first_graph.add_edge(node_1, node_2)

    print(first_graph.number_of_nodes())
    print(first_graph.number_of_edges())

    second_graph = nx.Graph()
    fn = 'higgs-mention_network.edgelist'
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            terms = line.strip().split(' ')
            node_1, node_2 = terms[0], terms[1]
            second_graph.add_edge(node_1, node_2)

    fn = 'higgs-retweet_network.edgelist'
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            terms = line.strip().split(' ')
            node_1, node_2 = terms[0], terms[1]
            second_graph.add_edge(node_1, node_2)

    fn = 'higgs-reply_network.edgelist'
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            terms = line.strip().split(' ')
            node_1, node_2 = terms[0], terms[1]
            second_graph.add_edge(node_1, node_2)

    print(second_graph.number_of_nodes())
    print(second_graph.number_of_edges())

    first_graph_node_set = set([node for node in first_graph.nodes()])
    second_graph_node_set = set([node for node in second_graph.nodes()])
    combine_node_set = first_graph_node_set & second_graph_node_set
    print('combin set', len(combine_node_set))

    lcc = max(nx.connected_component_subgraphs(first_graph), key=len)

    print('max lcc first graph', len(lcc))
    first_subgraph = nx.subgraph(first_graph, lcc)
    print(first_subgraph.number_of_nodes())
    print(first_subgraph.number_of_edges())
    print(nx.is_connected(first_subgraph))

    second_subgraph = nx.subgraph(second_graph, lcc)
    print(second_subgraph.number_of_nodes())
    print(second_subgraph.number_of_edges())
    print(nx.is_connected(second_subgraph))

    lcc = max(nx.connected_component_subgraphs(nx.subgraph(first_graph, combine_node_set)), key=len)

    print('max lcc combine set in first graph', len(lcc))
    first_subgraph = nx.subgraph(first_graph, lcc)
    print(first_subgraph.number_of_nodes())
    print(first_subgraph.number_of_edges())
    print(nx.is_connected(first_subgraph))

    second_subgraph = nx.subgraph(second_graph, lcc)
    print(second_subgraph.number_of_nodes())
    print(second_subgraph.number_of_edges())
    print(nx.is_connected(second_subgraph))









if __name__ == '__main__':

    generate_dataset()