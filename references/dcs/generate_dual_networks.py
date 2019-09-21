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

def generate_instance(case_id, verbose=True):
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
        normalized_prob_dist = [prob / float(sum_prob) if not sum_prob == 0. else 1. / len(neighbors) for prob in neighbor_degree_dist]

        if np.random.uniform() > restart: # random walk to next node
            next_node = np.random.choice(neighbors,
                                         p=normalized_prob_dist)  # biased for those nodes with high degree
        else:  # restart from initial node
            next_node = start_node

        subgraph.add(next_node)
        if verbose:
            print('generating {:d} nodes ...'.format(len(subgraph)))

    # generate node weight
    mean_1 = 3
    mean_2 = 0.
    std = 1.
    weight = np.zeros(num_nodes)
    for node in first_graph.nodes():
        if node in subgraph:
            weight[node] = np.random.normal(mean_1, std)
        else:
            weight[node] = np.random.normal(mean_2, std)

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic'
    # path = '/Users/fei/Downloads/Tmp/dcs_syn_dataset/data'
    fn = 'dual_mu_{}_nodes_{}_deg1_{}_deg2_{}_subsize_{}_restart_{}_{}.pkl'.format(mean_1, num_nodes, first_deg, second_deg, subgraph_size, restart, case_id)

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


def check_instance():
    num_nodes = 1000
    first_deg = 3
    second_deg = 10
    subgraph_size = 100
    case_id = 2

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic'
    # path = '/Users/fei/Downloads/Tmp/dcs_syn_dataset/data'
    fn = 'dual_nodes_{}_deg1_{}_deg2_{}_subsize_{}_{}.pkl'.format(num_nodes, first_deg, second_deg, subgraph_size, case_id)
    # fn = 'dual_nodes_{}_deg1_{}_deg2_{}_subsize_{}.pkl'.format(num_nodes, first_deg, second_deg, subgraph_size)


    with open(os.path.join(path, fn), 'rb') as rfile:
        instance = pickle.load(rfile)

    first_graph = instance['first_graph']
    second_graph = instance['second_graph']
    subgraph = instance['true_subgraph']

    # x = max(nx.connected_component_subgraphs(second_graph), key=len)
    # y = nx.diameter(nx.subgraph(second_graph, x))
    # print('diameter = {}'.format(y))

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


def generate_dcs_dataset(id):
    num_nodes = 1000
    first_deg = 3
    second_deg = 10
    subgraph_size = 100
    mu = 3
    case_id = id
    restart = 0.3
    rpath = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic'
    # wpath = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic/dcs/mu_{}/case_{}'.format(mu, case_id)
    wpath = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic/dcs/mu{}_restart{}/case_{}'.format(mu, restart, case_id)

    # fn = 'dual_nodes_1000_deg1_3_deg2_10_subsize_100_{}.pkl'.format(case_id)
    fn = 'dual_mu_{}_nodes_{}_deg1_{}_deg2_{}_subsize_{}_restart_{}_{}.pkl'.format(mu, num_nodes, first_deg, second_deg, subgraph_size, restart, case_id)
    with open(os.path.join(rpath, fn), 'rb') as rfile:
        instance = pickle.load(rfile)

    first_graph = instance['first_graph']
    second_graph = instance['second_graph']

    if first_graph.number_of_nodes() != second_graph.number_of_nodes():
        print('error !!!')

    fn = '01Nodes.txt'
    with open(os.path.join(wpath, fn), 'w') as wfile:
        for node in first_graph.nodes():
            wfile.write('{}\n'.format(node))

    # check 01Nodes, is nodes be sorted in order ascendendingly
    fn = '02EdgesP.txt'
    with open(os.path.join(wpath, fn), 'w') as wfile:
        for edge in first_graph.edges():
            wfile.write('{} {}\n'.format(edge[0], edge[1]))

    fn = '03EdgesC.txt'
    with open(os.path.join(wpath, fn), 'w') as wfile:
        for edge in second_graph.edges():
            wfile.write('{} {} 1.0\n'.format(edge[0], edge[1]))

    weight = instance['weight']
    fn = 'sorted_attributes.txt'
    with open(os.path.join(wpath, fn), 'w') as wfile:
        x = np.argsort(weight)[::-1]
        for i in x:
            wfile.write('{} {:.5f}\n'.format(i, weight[i]))


def remove_and_generate(id):
    # remove top subgraph and generate new files to run dcs
    # rpath = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic/dcs/top_not_remove/case_{}'.format(id)
    rpath = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Mus_Multiplex_Genetic/dcs/top_not_remove/case_{}'.format(id) # mus dataset
    # rpath = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic/dcs/top_remove/case_{}'.format(id)
    # wpath = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic/dcs/top_remove/case_{}'.format(id)
    wpath = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Mus_Multiplex_Genetic/dcs/top_remove/case_{}'.format(id)
    # wpath = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic/dcs/top_remove_2/case_{}'.format(id)

    # get nodes to be removed
    fn = 'Output_top1/01DenseSubgraphNodes.txt' # note, Output_top1
    # fn = 'Output/01DenseSubgraphNodes.txt'
    subgraph = set()
    with open(os.path.join(rpath, fn)) as rfile:
        for line in rfile:
            node = int(line.strip())
            subgraph.add(node)

    # get nodes where remove from
    fn = '01Nodes.txt'
    graph_before_removed = set()
    with open(os.path.join(rpath, fn)) as rfile:
        for line in rfile:
            node = int(line.strip())
            graph_before_removed.add(node)

    graph_before_removed = sorted(list(graph_before_removed))

    # nodes after removed
    # node file
    # fn = '01Nodes.txt'
    graph_after_removed = set()
    # with open(os.path.join(wpath, fn), 'w') as wfile:
    for node in graph_before_removed:
        if node in subgraph:
            continue
        else:
            # wfile.write('{}\n'.format(node))
            graph_after_removed.add(node)

    # sorted
    graph_after_removed = sorted(list(graph_after_removed))


    # load graph file
    # path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic'
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Mus_Multiplex_Genetic/dual' # mus dataset
    # fn = 'dual_nodes_1000_deg1_3_deg2_10_subsize_100_{}.pkl'.format(id)
    fn = 'dual_mus_case_{}.pkl'.format(id)
    with open(os.path.join(path, fn), 'rb') as rfile:
        instance = pickle.load(rfile)

    first_graph = instance['first_graph']
    second_graph = instance['second_graph']
    weight = instance['weight']

    # new graph induced by remaining nodes
    whole_first_graph_after_removed = nx.subgraph(first_graph, graph_after_removed)

    # print(len([x for x in nx.connected_components(first_graph_after_removed)]))

    # second_graph_after_removed = nx.subgraph(second_graph, graph_after_removed)

    # print(first_graph_after_removed.number_of_nodes())
    # print(second_graph_after_removed.number_of_nodes())

    # all component after removed some nodes
    components = [x for x in nx.connected_components(whole_first_graph_after_removed)]
    print(len(components))

    # generate files for each component
    for i, component in enumerate(components):
        # for each connected component

        sorted_component = sorted(list(component))
        map_node = {}
        for node_id, node in enumerate(sorted_component):
            map_node[node] = node_id

        fn = '01Nodes_{}.txt'.format(i)
        with open(os.path.join(wpath, fn), 'w') as wfile:
            for node in sorted(list(component)):
                wfile.write('{}\n'.format(node))

        first_graph_after_removed = nx.subgraph(first_graph, sorted_component)

        # physical file
        fn = '02EdgesP_{}.txt'.format(i)
        component_first_graph = nx.subgraph(first_graph, component)
        with open(os.path.join(wpath, fn), 'w') as wfile:
            for edge in component_first_graph.edges(): # first graph
                node_1, node_2 = edge
                wfile.write('{} {}\n'.format(map_node[node_1], map_node[node_2]))

        second_graph_after_removed = nx.subgraph(second_graph, sorted_component)

        print(first_graph_after_removed.number_of_nodes())
        print(second_graph_after_removed.number_of_nodes())
        # conceptual file
        fn = '03EdgesC_{}.txt'.format(i)
        with open(os.path.join(wpath, fn), 'w') as wfile:
            for edge in second_graph_after_removed.edges(): # second graph
                node_1, node_2 = edge
                wfile.write('{} {} 1.0\n'.format(map_node[node_1], map_node[node_2]))

        # new id, must sorted component, or error
        seed = sorted_component[np.argmax(weight[sorted_component])]
        print('max id {}, max val {}'.format(np.argmax(weight[sorted_component]), np.max(weight[sorted_component])))
        fn = '04Seeds_{}.txt'.format(i)
        with open(os.path.join(wpath, fn), 'w') as wfile:
            wfile.write('{}'.format(seed))

        # if i == 0:
        #     for edge in first_graph_after_removed.edges():
        #         print(edge)
        #     print('vvvvv')
        #     for edge in second_graph_after_removed.edges():
        #         print(edge)


def remove_and_generate_2(id):
    # remove top subgraph and generate new files to run dcs
    # rpath = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic/dcs/top_remove/case_{}'.format(id)
    rpath = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Mus_Multiplex_Genetic/dcs/top_remove/case_{}'.format(id)
    # wpath = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic/dcs/top_remove_2/case_{}'.format(id)
    wpath = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Mus_Multiplex_Genetic/dcs/top_remove_2/case_{}'.format(id)

    fn = 'Output/01DenseSubgraphNodes.txt'
    subgraph = set()
    with open(os.path.join(rpath, fn)) as rfile:
        for line in rfile:
            node = int(line.strip())
            subgraph.add(node)

    component_id = 0

    fn = '01Nodes_{}.txt'.format(component_id)
    graph_before_removed = set()
    with open(os.path.join(rpath, fn)) as rfile:
        for line in rfile:
            node = int(line.strip())
            graph_before_removed.add(node)

    graph_before_removed = sorted(list(graph_before_removed))

    # node file
    # fn = '01Nodes.txt'
    graph_after_removed = set()
    # with open(os.path.join(wpath, fn), 'w') as wfile:
    for node in graph_before_removed:
        if node in subgraph:
            continue
        else:
            # wfile.write('{}\n'.format(node))
            graph_after_removed.add(node)

    graph_after_removed = sorted(list(graph_after_removed))


    # load graph file
    # path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic'
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Mus_Multiplex_Genetic/dual'  # mus dataset
    # fn = 'dual_nodes_1000_deg1_3_deg2_10_subsize_100_{}.pkl'.format(id)
    fn = 'dual_mus_case_{}.pkl'.format(id)
    with open(os.path.join(path, fn), 'rb') as rfile:
        instance = pickle.load(rfile)

    first_graph = instance['first_graph']
    second_graph = instance['second_graph']
    weight = instance['weight']

    whole_first_graph_after_removed = nx.subgraph(first_graph, graph_after_removed)

    # print(len([x for x in nx.connected_components(first_graph_after_removed)]))

    # second_graph_after_removed = nx.subgraph(second_graph, graph_after_removed)

    # print(first_graph_after_removed.number_of_nodes())
    # print(second_graph_after_removed.number_of_nodes())

    components = [x for x in nx.connected_components(whole_first_graph_after_removed)]
    print(len(components))

    for i, component in enumerate(components):
        # for each connected component

        print(len(component))

        sorted_component = sorted(list(component))
        map_node = {}
        for node_id, node in enumerate(sorted_component):
            map_node[node] = node_id

        fn = '01Nodes_{}.txt'.format(i)
        with open(os.path.join(wpath, fn), 'w') as wfile:
            for node in sorted(list(component)):
                wfile.write('{}\n'.format(node))

        first_graph_after_removed = nx.subgraph(first_graph, sorted_component)

        # physical file
        fn = '02EdgesP_{}.txt'.format(i)
        component_first_graph = nx.subgraph(first_graph, component)
        with open(os.path.join(wpath, fn), 'w') as wfile:
            for edge in component_first_graph.edges(): # first graph
                node_1, node_2 = edge
                wfile.write('{} {}\n'.format(map_node[node_1], map_node[node_2]))

        second_graph_after_removed = nx.subgraph(second_graph, sorted_component)

        print(first_graph_after_removed.number_of_nodes())
        print(second_graph_after_removed.number_of_nodes())
        # conceptual file
        fn = '03EdgesC_{}.txt'.format(i)
        with open(os.path.join(wpath, fn), 'w') as wfile:
            for edge in second_graph_after_removed.edges(): # second graph
                node_1, node_2 = edge
                wfile.write('{} {} 1.0\n'.format(map_node[node_1], map_node[node_2]))

        seed = sorted_component[np.argmax(weight[sorted_component])]
        print('max id {}, max val {}'.format(np.argmax(weight[sorted_component]), np.max(weight[sorted_component])))
        fn = '04Seeds_{}.txt'.format(i)
        with open(os.path.join(wpath, fn), 'w') as wfile:
            wfile.write('{}'.format(seed))

        if i == 2:
            for edge in first_graph_after_removed.edges():
                print(edge)
            print('vvvvv')
            for edge in second_graph_after_removed.edges():
                print(edge)

# def check_stat(case_id):
#     path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic'
#     fn = 'dual_mus_case_{}.pkl'.format(case_id)
#     with open(os.path.join(path, fn), 'rb') as rfile:
#         instance = pickle.load(rfile)
#
#     first_graph = instance['first_graph']
#     second_graph = instance['second_graph']
#
#     if first_graph.number_of_nodes() != second_graph.number_of_nodes():
#         print('error !!!')
#
#     num_nodes = first_graph.number_of_nodes()
#     subgraph = instance['true_subgraph']
#
#     print('num of nodes: {:d}'.format(num_nodes))
#     print('num of edges: {:d} in first graph'.format(first_graph.number_of_edges()))
#     print('num of edges: {:d} in second graph'.format(second_graph.number_of_edges()))
#
#     density = nx.density(first_graph)
#     print('first graph density', density)
#
#     density = nx.density(second_graph)
#     print('second graph density', density)
#
#     first_subgraph = nx.subgraph(first_graph, subgraph)
#     density = nx.density(first_subgraph)
#     print('subgraph density in first graph', density)
#
#     second_subgraph = nx.subgraph(second_graph, subgraph)
#     density = nx.density(second_subgraph)
#     print('subgraph density in second graph', density)


if __name__ == '__main__':
    pass

    # for i in range(10):
    #     generate_instance(i)

    # check_instance()

    # x = [2, 1, 3, 5, 4]
    # y = np.argsort(x)[::-1]
    # print(y)

    for i in range(10):
        generate_dcs_dataset(i)

    # remove_and_generate(9)

    # remove_and_generate_2(9)