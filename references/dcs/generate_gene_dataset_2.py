#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : generate_dual_network_2
# @Date     : 09/13/2019 15:52:40
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


from __future__ import print_function


import os
import pickle

import numpy as np
import networkx as nx


def generate_homo_dataset():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Homo_Multiplex_Genetic/Dataset'
    fn = 'homo_genetic_multiplex.edges'
    first_graph = nx.Graph()
    second_graph = nx.Graph()

    graph_list = [nx.Graph() for _ in range(7)]
    with open(os.path.join(path, fn)) as rfile:
        count = 0
        for line in rfile:
            if line.startswith('1'):
                terms = line.strip().split(' ')
                node_1 = terms[1]
                node_2 = terms[2]
                graph_list[0].add_edge(node_1, node_2)
            elif line.startswith('2'):
                terms = line.strip().split(' ')
                node_1 = terms[1]
                node_2 = terms[2]
                graph_list[1].add_edge(node_1, node_2)
            elif line.startswith('3'):
                terms = line.strip().split(' ')
                node_1 = terms[1]
                node_2 = terms[2]
                graph_list[2].add_edge(node_1, node_2)
            elif line.startswith('4'):
                terms = line.strip().split(' ')
                node_1 = terms[1]
                node_2 = terms[2]
                graph_list[3].add_edge(node_1, node_2)
            elif line.startswith('5'):
                terms = line.strip().split(' ')
                node_1 = terms[1]
                node_2 = terms[2]
                graph_list[4].add_edge(node_1, node_2)
            elif line.startswith('6'):
                terms = line.strip().split(' ')
                node_1 = terms[1]
                node_2 = terms[2]
                graph_list[5].add_edge(node_1, node_2)
            elif line.startswith('7'):
                terms = line.strip().split(' ')
                node_1 = terms[1]
                node_2 = terms[2]
                graph_list[6].add_edge(node_1, node_2)

            else:
                continue

    print('Direct interaction', graph_list[0].number_of_nodes(), graph_list[0].number_of_edges())
    print('Physical association', graph_list[1].number_of_nodes(), graph_list[1].number_of_edges())
    print('Suppressive genetic interaction defined by inequality', graph_list[2].number_of_nodes(), graph_list[2].number_of_edges())
    print('Association', graph_list[3].number_of_nodes(), graph_list[3].number_of_edges())
    print('Colocalization', graph_list[4].number_of_nodes(), graph_list[4].number_of_edges())
    print('Additive genetic interaction defined by inequality', graph_list[5].number_of_nodes(), graph_list[5].number_of_edges())
    print('Synthetic genetic interaction defined by inequality', graph_list[6].number_of_nodes(), graph_list[6].number_of_edges())

    first_layer = 0
    second_layer = 4

    first_graph_node_set = set([node for node in graph_list[first_layer]])
    second_graph_node_set = set([node for node in graph_list[second_layer]])
    combine_node_set = first_graph_node_set & second_graph_node_set
    print('combine node set', len(combine_node_set))

    first_graph = nx.subgraph(graph_list[first_layer], combine_node_set)
    second_graph = nx.subgraph(graph_list[second_layer], combine_node_set)

    print(first_graph.number_of_nodes())
    print(first_graph.number_of_edges())
    print(nx.is_connected(first_graph))
    print(second_graph.number_of_nodes())
    print(second_graph.number_of_edges())
    print(nx.is_connected(second_graph))

    print('-' * 10)
    lcc = max(nx.connected_component_subgraphs(second_graph), key=len)
    print(len(lcc))

    first_subgraph = nx.subgraph(first_graph, lcc)
    print(first_subgraph.number_of_nodes())
    print(first_subgraph.number_of_edges())
    print(nx.is_connected(first_subgraph))

    second_subgraph = nx.subgraph(second_graph, lcc)
    print(second_subgraph.number_of_nodes())
    print(second_subgraph.number_of_edges())
    print(nx.is_connected(second_subgraph))

    print('-' * 10)
    lcc = max(nx.connected_component_subgraphs(first_graph), key=len)
    print(len(lcc))

    first_subgraph = nx.subgraph(first_graph, lcc)
    print(first_subgraph.number_of_nodes())
    print(first_subgraph.number_of_edges())
    print(nx.is_connected(first_subgraph))

    second_subgraph = nx.subgraph(second_graph, lcc)
    print(second_subgraph.number_of_nodes())
    print(second_subgraph.number_of_edges())
    print(nx.is_connected(second_subgraph))

    print('-' * 10)
    lcc = max(nx.connected_component_subgraphs(second_subgraph), key=len)
    print(len(lcc))

    first_subgraph = nx.subgraph(first_graph, lcc)
    print(first_subgraph.number_of_nodes())
    print(first_subgraph.number_of_edges())
    print(nx.is_connected(first_subgraph))

    second_subgraph = nx.subgraph(second_graph, lcc)
    print(second_subgraph.number_of_nodes())
    print(second_subgraph.number_of_edges())
    print(nx.is_connected(second_subgraph))

    print('-' * 10)
    lcc = max(nx.connected_component_subgraphs(second_subgraph), key=len)
    print(len(lcc))

    first_subgraph = nx.subgraph(first_graph, lcc)
    print(first_subgraph.number_of_nodes())
    print(first_subgraph.number_of_edges())
    print(nx.is_connected(first_subgraph))

    second_subgraph = nx.subgraph(second_graph, lcc)
    print(second_subgraph.number_of_nodes())
    print(second_subgraph.number_of_edges())
    print(nx.is_connected(second_subgraph))

    print('-' * 10)
    lcc = max(nx.connected_component_subgraphs(first_subgraph), key=len)
    print(len(lcc))

    first_subgraph = nx.subgraph(first_graph, lcc)
    print(first_subgraph.number_of_nodes())
    print(first_subgraph.number_of_edges())
    print(nx.is_connected(first_subgraph))

    second_subgraph = nx.subgraph(second_graph, lcc)
    print(second_subgraph.number_of_nodes())
    print(second_subgraph.number_of_edges())
    print(nx.is_connected(second_subgraph))

    print('-' * 10)
    lcc = max(nx.connected_component_subgraphs(second_subgraph), key=len)
    print(len(lcc))

    first_subgraph = nx.subgraph(first_graph, lcc)
    print(first_subgraph.number_of_nodes())
    print(first_subgraph.number_of_edges())
    print(nx.is_connected(first_subgraph))

    second_subgraph = nx.subgraph(second_graph, lcc)
    print(second_subgraph.number_of_nodes())
    print(second_subgraph.number_of_edges())
    print(nx.is_connected(second_subgraph))

    print('test')

    first_graph = nx.subgraph(graph_list[first_layer], combine_node_set)
    second_graph = nx.subgraph(graph_list[second_layer], combine_node_set)
    first_subgraph = nx.subgraph(first_graph, lcc)
    print(first_subgraph.number_of_nodes())
    print(first_subgraph.number_of_edges())
    print(nx.is_connected(first_subgraph))

    second_subgraph = nx.subgraph(second_graph, lcc)
    print(second_subgraph.number_of_nodes())
    print(second_subgraph.number_of_edges())
    print(nx.is_connected(second_subgraph))


    # print('first graph')
    # for node_1, node_2 in first_subgraph.edges():
    #     print(node_1, node_2)

    # print('second graph')
    # for node_1, node_2 in second_subgraph.edges():
    #     print(node_1, node_2)

    instance = {}
    instance['first_graph'] = second_subgraph # note, first <-> second instance, second <-> first instance


    instance['second_graph'] = first_subgraph

    print(instance['first_graph'].number_of_nodes())
    print(instance['first_graph'].number_of_edges())
    print(nx.is_connected(instance['first_graph']))
    print(instance['second_graph'].number_of_nodes())
    print(instance['second_graph'].number_of_edges())
    print(nx.is_connected(instance['second_graph']))

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Homo_Multiplex_Genetic/dual'
    fn = 'structures.pkl'
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(instance, wfile)

def generate_mus_dataset():
    pass


def relabel_graph():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Homo_Multiplex_Genetic/dual'
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
        node_map[node] = i
        tmp_node_map[node] = str(node)
        inverse_node_map[i] = node

    tmp_relabeled_first_graph = nx.relabel_nodes(first_graph, tmp_node_map)
    tmp_relabeled_second_graph = nx.relabel_nodes(second_graph, tmp_node_map)

    new_node_map = {}
    for k, v in node_map.items():
        new_node_map[str(k)] = v

    relabeled_first_graph = nx.relabel_nodes(tmp_relabeled_first_graph, new_node_map)
    relabled_second_graph = nx.relabel_nodes(tmp_relabeled_second_graph, new_node_map)

    dataset = {}
    dataset['first_graph'] = relabeled_first_graph
    dataset['second_graph'] = relabled_second_graph
    dataset['node_map'] = node_map
    dataset['inverse_node_map'] = inverse_node_map
    fn = 'relabeled_structures.pkl'
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(dataset, wfile)


def generate_attributes(case_id):
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Homo_Multiplex_Genetic/dual'
    fn = 'relabeled_structures.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        dataset = pickle.load(rfile)

    first_graph = dataset['first_graph']
    second_graph = dataset['second_graph']

    num_nodes = first_graph.number_of_nodes()
    if first_graph.number_of_nodes() != second_graph.number_of_nodes():
        print('error !')
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
    subgraph_size = 500
    start_node = next_node = np.random.choice(range(num_nodes))
    subgraph = set()
    subgraph.add(start_node)

    while True:
        if len(subgraph) >= subgraph_size:
            break

        neighbors = [node for node in nx.neighbors(first_graph, next_node)]
        neighbor_degree_dist = [second_graph_node_degree[node] for node in neighbors]
        sum_prob = np.sum(neighbor_degree_dist)
        normalized_prob_dist = [prob / sum_prob if not sum_prob == 0. else 1. / len(neighbors) for prob in neighbor_degree_dist]

        if np.random.uniform() > restart:
            next_node = np.random.choice(neighbors, p=normalized_prob_dist)
        else:
            next_node = start_node

        subgraph.add(next_node)
        print('generating {:d} nodes ...'.format(len(subgraph)))

    # generate node weight
    mean_1 = 3.
    mean_2 = 0.
    std = 1.
    weight = np.zeros(num_nodes)
    for node in first_graph.nodes:
        if node in subgraph:
            weight[node] = np.random.normal(mean_1, std)
        else:
            weight[node] = np.random.normal(mean_2, std)

    fn = 'dual_mu_{}_case_{}.pkl'.format(mean_1, case_id)

    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump({'first_graph': first_graph, 'second_graph': second_graph, 'true_subgraph': subgraph, 'weight': weight}, wfile)


    print('num of nodes: {:d}'.format(num_nodes))
    print('num of edges: {:d}'.format(first_graph.number_of_edges()))
    print('first graph connected', nx.is_connected(first_graph))
    print('num of edges: {:d}'.format(second_graph.number_of_edges()))
    print('second graph connected', nx.is_connected(second_graph))

    density = nx.density(first_graph)
    print('first graph density', density)

    density = nx.density(second_graph)
    print('second graph density', density)

    first_subgraph = nx.subgraph(first_graph, subgraph)
    density = nx.density(first_subgraph)
    print('true subgraph density in first graph', density)

    second_subgraph = nx.subgraph(second_graph, subgraph)
    density = nx.density(second_subgraph)
    print('true subgraph density in second graph', density)




if __name__ == '__main__':
    pass

    # generate_homo_dataset()

    # relabel_graph()

    for case_id in range(5, 10):
        generate_attributes(case_id=case_id)