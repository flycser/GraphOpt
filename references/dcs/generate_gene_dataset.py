#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : generate_gene_dataset
# @Date     : 08/04/2019 18:17:22
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


from __future__ import print_function

import os
import pickle

import numpy as np
import networkx as nx


def generate_Arabidopsis_dataset():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Arabidopsis_Multiplex_Genetic/Dataset'
    fn = 'arabidopsis_genetic_multiplex.edges'
    first_graph = nx.Graph()
    second_graph = nx.Graph()
    with open(os.path.join(path, fn)) as rfile:
        count = 0
        for line in rfile:
            if line.startswith('1'):
                terms = line.strip().split(' ')
                node_1 = terms[1]
                node_2 = terms[2]
                first_graph.add_edge(node_1, node_2)
                count += 1
            elif line.startswith('2'):
                terms = line.strip().split(' ')
                node_1 = terms[1]
                node_2 = terms[2]
                second_graph.add_edge(node_1, node_2)
                count += 1
            else:
                continue

        print(count)

    print(first_graph.number_of_nodes())
    print(first_graph.number_of_edges())
    print(second_graph.number_of_nodes())
    print(second_graph.number_of_edges())

    first_graph_node_set = set([node for node in first_graph.nodes()])
    second_graph_node_set = set([node for node in second_graph.nodes()])
    combine_set = first_graph_node_set & second_graph_node_set
    print(len(combine_set))

    first_subgraph = nx.subgraph(first_graph, combine_set)
    second_subgraph = nx.subgraph(second_graph, combine_set)

    print(first_subgraph.number_of_nodes())
    print(first_subgraph.number_of_edges())
    print(nx.is_connected(first_subgraph))
    print(second_subgraph.number_of_nodes())
    print(second_subgraph.number_of_edges())
    print(nx.is_connected(second_subgraph))

    lcc = max(nx.connected_component_subgraphs(second_subgraph), key=len)

    print(len(lcc))

    subgraph = nx.subgraph(second_graph, lcc)
    print(subgraph.number_of_nodes())
    print(subgraph.number_of_edges())
    print(nx.is_connected(subgraph))

    subgraph = nx.subgraph(first_graph, lcc)
    print(subgraph.number_of_nodes())
    print(subgraph.number_of_edges())
    print(nx.is_connected(subgraph))

    lcc = max(nx.connected_component_subgraphs(first_subgraph), key=len)

    print(len(lcc))

    subgraph = nx.subgraph(second_graph, lcc)
    print(subgraph.number_of_nodes())
    print(subgraph.number_of_edges())
    print(nx.is_connected(subgraph))

    instance = {}
    instance['first_graph'] = subgraph

    subgraph = nx.subgraph(first_graph, lcc)
    print(subgraph.number_of_nodes())
    print(subgraph.number_of_edges())
    print(nx.is_connected(subgraph))

    instance['second_graph'] = subgraph

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Arabidopsis_Multiplex_Genetic/dual'
    fn = 'structures.pkl'
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(instance, wfile)



def generate_Homo_dataset():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Homo_Multiplex_Genetic/Dataset'
    fn = 'homo_genetic_multiplex.edges'
    first_graph = nx.Graph()
    second_graph = nx.Graph()
    with open(os.path.join(path, fn)) as rfile:
        count = 0
        for line in rfile:
            if line.startswith('1'):
                terms = line.strip().split(' ')
                node_1 = terms[1]
                node_2 = terms[2]
                first_graph.add_edge(node_1, node_2)
                count += 1
            elif line.startswith('2'):
                terms = line.strip().split(' ')
                node_1 = terms[1]
                node_2 = terms[2]
                second_graph.add_edge(node_1, node_2)
                count += 1
            else:
                continue

        print(count)

    print(first_graph.number_of_nodes())
    print(first_graph.number_of_edges())
    print(second_graph.number_of_nodes())
    print(second_graph.number_of_edges())

    first_graph_node_set = set([node for node in first_graph.nodes()])
    second_graph_node_set = set([node for node in second_graph.nodes()])
    combine_set = first_graph_node_set & second_graph_node_set
    print(len(combine_set))

    first_subgraph = nx.subgraph(first_graph, combine_set)
    second_subgraph = nx.subgraph(second_graph, combine_set)

    print(first_subgraph.number_of_nodes())
    print(first_subgraph.number_of_edges())
    print(nx.is_connected(first_subgraph))
    print(second_subgraph.number_of_nodes())
    print(second_subgraph.number_of_edges())
    print(nx.is_connected(second_subgraph))

    lcc = max(nx.connected_component_subgraphs(second_subgraph), key=len)

    print(len(lcc))

    subgraph = nx.subgraph(second_graph, lcc)
    print(subgraph.number_of_nodes())
    print(subgraph.number_of_edges())
    print(nx.is_connected(subgraph))

    subgraph = nx.subgraph(first_graph, lcc)
    print(subgraph.number_of_nodes())
    print(subgraph.number_of_edges())
    print(nx.is_connected(subgraph))

    lcc = max(nx.connected_component_subgraphs(first_subgraph), key=len)

    print(len(lcc))

    subgraph = nx.subgraph(second_graph, lcc)
    print(subgraph.number_of_nodes())
    print(subgraph.number_of_edges())
    print(nx.is_connected(subgraph))

    instance = {}
    instance['first_graph'] = subgraph

    subgraph = nx.subgraph(first_graph, lcc)
    print(subgraph.number_of_nodes())
    print(subgraph.number_of_edges())
    print(nx.is_connected(subgraph))

    instance['second_graph'] = subgraph

    # path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Homo_Multiplex_Genetic/dual'
    # fn = 'structures.pkl'
    # with open(os.path.join(path, fn), 'wb') as wfile:
    #     pickle.dump(instance, wfile)


def generate_Mus_dataset():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Mus_Multiplex_Genetic/Dataset'
    fn = 'mus_genetic_multiplex.edges'
    first_graph = nx.Graph()
    second_graph = nx.Graph()
    with open(os.path.join(path, fn)) as rfile:
        count = 0
        for line in rfile:
            if line.startswith('3'):
                terms = line.strip().split(' ')
                node_1 = int(terms[1])
                node_2 = int(terms[2])
                first_graph.add_edge(node_1, node_2)
                count += 1
            elif line.startswith('1'):
                terms = line.strip().split(' ')
                node_1 = int(terms[1])
                node_2 = int(terms[2])
                second_graph.add_edge(node_1, node_2)
                count += 1
            else:
                continue

        print(count)

    print(first_graph.number_of_nodes())
    print(first_graph.number_of_edges())
    print(second_graph.number_of_nodes())
    print(second_graph.number_of_edges())

    print('combine')

    first_graph_node_set = set([node for node in first_graph.nodes()])
    second_graph_node_set = set([node for node in second_graph.nodes()])
    combine_set = first_graph_node_set & second_graph_node_set
    print(len(combine_set))

    # interaction of two set
    first_subgraph = nx.subgraph(first_graph, combine_set)
    second_subgraph = nx.subgraph(second_graph, combine_set)

    print(first_subgraph.number_of_nodes())
    print(first_subgraph.number_of_edges())
    print(nx.is_connected(first_subgraph))
    print(second_subgraph.number_of_nodes())
    print(second_subgraph.number_of_edges())
    print(nx.is_connected(second_subgraph))

    print('physical')

    # largrest comnnected component
    lcc = max(nx.connected_component_subgraphs(second_subgraph), key=len)

    print(len(lcc))

    subgraph = nx.subgraph(second_graph, lcc)
    print(subgraph.number_of_nodes())
    print(subgraph.number_of_edges())
    print(nx.is_connected(subgraph))

    subgraph = nx.subgraph(first_graph, lcc)
    print(subgraph.number_of_nodes())
    print(subgraph.number_of_edges())
    print(nx.is_connected(subgraph))

    print('interaction')

    lcc = max(nx.connected_component_subgraphs(first_subgraph), key=len)

    print(len(lcc))

    subgraph = nx.subgraph(second_graph, lcc)
    print(subgraph.number_of_nodes())
    print(subgraph.number_of_edges())
    print(nx.is_connected(subgraph))

    instance = {}
    instance['second_graph'] = subgraph
    # lcc = max(nx.connected_component_subgraphs(subgraph), key=len)
    # print(len(lcc))

    subgraph = nx.subgraph(first_graph, lcc)
    print(subgraph.number_of_nodes())
    print(subgraph.number_of_edges())
    print(nx.is_connected(subgraph))

    instance['first_graph'] = subgraph

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Mus_Multiplex_Genetic/dual'
    fn = 'structures.pkl'
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(instance, wfile)


def relabel_graph():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Mus_Multiplex_Genetic/dual'
    fn = 'structures.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        dataset = pickle.load(rfile)

    first_graph = dataset['first_graph']
    second_graph = dataset['second_graph']

    node_map = {}
    tmp_node_map = {}
    inverse_node_map = {}
    sorted_nodes = sorted([node for node in first_graph.nodes()])
    print(sorted_nodes)
    print(len(sorted_nodes))

    for i, node in enumerate(sorted_nodes):
        node_map[node] = i # old node id, new node id
        tmp_node_map[node] = str(node) # old node id, string old node id
        inverse_node_map[i] = node # new node id, old node id

    # import operator

    # sorted_x = sorted(node_map.items(), key=operator.itemgetter(1))
    # print(sorted_x)

    # for edge in first_graph.edges():
        # print(edge)

    # print(node_map)

    tmp_relabeled_first_graph = nx.relabel_nodes(first_graph, tmp_node_map) # note, same type may result in error !!!
    tmp_relabeled_second_graph = nx.relabel_nodes(second_graph, tmp_node_map)

    new_node_map = {}
    for k, v in node_map.items():
        new_node_map[str(k)] = v

    # sorted_x = sorted(new_node_map.items(), key=operator.itemgetter(1))
    # print(sorted_x)

    relabeled_first_graph = nx.relabel_nodes(tmp_relabeled_first_graph, new_node_map)
    relabeled_second_graph = nx.relabel_nodes(tmp_relabeled_second_graph, new_node_map)

    # for edge in relabeled_first_graph.edges():
    #     print(edge)

    dataset = {}
    dataset['first_graph'] = relabeled_first_graph
    dataset['second_graph'] = relabeled_second_graph
    dataset['node_map'] = node_map
    dataset['inverse_node_map'] = inverse_node_map
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Mus_Multiplex_Genetic/dual'
    fn = 'relabeled_structures.pkl'
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(dataset, wfile)



def generate_attributes(case_id):
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Mus_Multiplex_Genetic/dual'
    fn = 'relabeled_structures.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        dataset = pickle.load(rfile)

    first_graph = dataset['first_graph']
    second_graph = dataset['second_graph']

    num_nodes = first_graph.number_of_nodes()
    if first_graph.number_of_nodes() != second_graph.number_of_nodes():
        print('error !!!')
        return

    print(nx.is_connected(first_graph))
    print(first_graph.number_of_nodes())
    print(first_graph.number_of_edges())
    print(nx.density(first_graph))
    print(nx.is_connected(second_graph))
    print(second_graph.number_of_nodes())
    print(second_graph.number_of_edges())
    print(nx.density(second_graph))

    second_graph_node_degree = np.zeros(num_nodes)
    for current_node in second_graph.nodes():
        second_graph_node_degree[current_node] = nx.degree(second_graph, current_node)

    restart = 0.3
    subgraph_size = 100
    start_node = next_node = np.random.choice(range(num_nodes))
    subgraph = set()
    subgraph.add(start_node)

    # print(len(list(nx.isolates(second_graph)))) # print isolated nodes
    while True:
        if len(subgraph) >= subgraph_size:
            break

        neighbors = [node for node in nx.neighbors(first_graph, next_node)]

        # for node in neighbors:
        #     if second_graph_node_degree[node] == 0.:
        #         print('v', node)
        #         print([node for node in nx.neighbors(second_graph, node)])
        #         print(len(list(nx.neighbors(second_graph, node))))

        neighbor_degree_dist = [second_graph_node_degree[node] for node in neighbors]

        # print(next_node)
        sum_prob = np.sum(neighbor_degree_dist)
        # print(neighbor_degree_dist)
        # print(sum_prob)

        normalized_prob_dist = [prob / sum_prob if not sum_prob == 0. else 1. / len(neighbors) for prob in neighbor_degree_dist] # note, if prob / sum_prob is float type
        # print(normalized_prob_dist)

        if np.random.uniform() > restart:
            next_node = np.random.choice(neighbors, p=normalized_prob_dist)
        else:
            next_node = start_node

        subgraph.add(next_node)
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

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Mus_Multiplex_Genetic/dual'
    fn = 'dual_mus_case_{}.pkl'.format(case_id)

    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump({'first_graph': first_graph, 'second_graph': second_graph, 'true_subgraph': subgraph, 'weight': weight}, wfile)

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

def check_mus_stat(case_id):
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Mus_Multiplex_Genetic/dual'
    fn = 'dual_mus_case_{}.pkl'.format(case_id)
    with open(os.path.join(path, fn), 'rb') as rfile:
        instance = pickle.load(rfile)

    first_graph = instance['first_graph']
    second_graph = instance['second_graph']

    if first_graph.number_of_nodes() != second_graph.number_of_nodes():
        print('error !!!')

    num_nodes = first_graph.number_of_nodes()
    subgraph = instance['true_subgraph']

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


def generate_mus_dcs(case_id):
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Mus_Multiplex_Genetic/dual'
    fn = 'dual_mus_case_{}.pkl'.format(case_id)
    with open(os.path.join(path, fn), 'rb') as rfile:
        instance = pickle.load(rfile)

    first_graph = instance['first_graph']
    second_graph = instance['second_graph']
    weight = instance['weight']

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Mus_Multiplex_Genetic/dcs/case_{}'.format(case_id)
    fn = '01Nodes.txt'
    sorted_nodes = sorted([node for node in first_graph.nodes()])
    with open(os.path.join(path, fn), 'w') as wfile:
        for node in sorted_nodes:
            wfile.write('{}\n'.format(node))

    fn = '02EdgesP.txt'
    with open(os.path.join(path, fn), 'w') as wfile:
        for edge in first_graph.edges():
            wfile.write('{} {}\n'.format(edge[0], edge[1]))

    fn = '03EdgesC.txt'
    with open(os.path.join(path, fn), 'w') as wfile:
        for edge in second_graph.edges():
            wfile.write('{} {} 1.0\n'.format(edge[0], edge[1]))

    fn = 'sorted_attributes.txt'
    with open(os.path.join(path, fn), 'w') as wfile:
        x = np.argsort(weight)[::-1]
        for i in x:
            wfile.write('{} {:.5f}\n'.format(i, weight[i]))




if __name__ == '__main__':

    # generate_Arabidopsis_dataset()

    # generate_Homo_dataset()

    # generate_Mus_dataset()

    # generate_attributes(9)

    # relabel_graph()

    check_mus_stat(2)

    # generate_mus_dcs(9)