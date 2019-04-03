#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : syn_evo
# @Date     : 03/30/2019 17:33:19
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


from __future__ import print_function

import os
import sys

sys.path.append(os.path.abspath(''))

import logging
from logging.config import fileConfig

fileConfig('../logging.conf')
# note: logger is not thread-safe, pls be careful when running multi-threads with logging
logger = logging.getLogger('fei')

import time
import pickle

import numpy as np
import networkx as nx


def grid_graph_structure(num_nodes):
    width = height = int(np.sqrt(num_nodes))
    graph = nx.grid_graph(dim=[width, height])

    # change node indices
    mapping = {}
    for (i, j) in graph.nodes():
        mapping[(i, j)] = i * width + j
    graph = nx.relabel_nodes(graph, mapping)
    logger.debug('-' * 5 + 'network info' + '-' * 5)
    logger.debug('number of nodes: {:d}'.format(nx.number_of_nodes(graph)))
    logger.debug('number of edges: {:d}'.format(nx.number_of_edges(graph)))
    # logger.debug('density: {:.2f}'.format(nx.density(graph)))
    #
    # all_deg = []
    # for idx, deg in nx.degree(graph ):
    #     all_deg.append(deg)
    # logger.debug('average degree: {:d}'.format(np.mean(all_deg)))
    # logger.debug('connected components in the graph: {:d}'.format(nx.number_connected_components(graph)))
    # largest_component = max(nx.connected_component_subgraphs(graph), key=len)
    # logger.debug('the largest component size: {:d}'.format(largest_component.number_of_nodes()))

    return graph

def generate_subgraphs(graph, num_time_stamps, num_nodes_subgraph_list, num_nodes_overlap_list):

    num_nodes = graph.number_of_nodes()
    subgraph_list = []

    # generate subgraph in first time stamp with random walk
    subgraph = []
    current_node = np.random.choice(num_nodes)
    while len(subgraph) < num_nodes_subgraph_list[0]:
        neighbors = list(nx.neighbors(graph, current_node))
        next_node = np.random.choice(neighbors)
        if next_node not in subgraph:
            subgraph.append(next_node)

        current_node = next_node

    subgraph_list.append(sorted(subgraph))

    for t in range(1, num_time_stamps):
        prev_subgraph = nx.subgraph(graph, subgraph)
        num_nodes_overlap = num_nodes_overlap_list[t - 1]
        num_nodes_subgraph = num_nodes_subgraph_list[t]
        if not 0 == num_nodes_subgraph and not 0 == len(prev_subgraph.nodes()): # two consecutive time stamps both exist signals
            subgraph = find_next_subgraph(graph, prev_subgraph, num_nodes_subgraph, num_nodes_overlap)
        elif not 0 == num_nodes_subgraph and 0 == len(prev_subgraph.nodes()): # previous time stamp does not exist signal
            subgraph = []
            current_node = np.random.choice(num_nodes)
            while len(subgraph) < num_nodes_subgraph:
                neighbors = list(nx.neighbors(graph, current_node))
                next_node = np.random.choice(neighbors)
                if next_node not in subgraph:
                    subgraph.append(next_node)

                current_node = next_node
        else:
            subgraph = []

        subgraph_list.append(sorted(subgraph))

        if not num_nodes_subgraph == 0:
            if nx.is_connected(nx.subgraph(graph, subgraph)):
                pass
            else:
                raise ('Disconnected Subgraph at time stamp {:d}'.format(t))

    return subgraph_list

def find_next_subgraph(graph, prev_subgraph, num_nodes_subgraph, num_nodes_overlap):
    logger.debug('number of nodes in previous subgraph: {:d}, number of nodes in current subgraph: {:d}, number of nodes overlapping: {:d}'.format(len(prev_subgraph.nodes()), num_nodes_subgraph, num_nodes_overlap))

    subgraph = []
    subgraph.append(np.random.choice(prev_subgraph.nodes()))
    while len(subgraph) < num_nodes_overlap:
        current_node = np.random.choice(subgraph)
        neighbors = list(nx.neighbors(prev_subgraph, current_node))
        next_node = np.random.choice(neighbors)
        if next_node not in subgraph:
            subgraph.append(next_node)

    while len(subgraph) < num_nodes_subgraph:
        current_node = np.random.choice(subgraph)
        neighbors = list(nx.neighbors(graph, current_node))
        next_node = np.random.choice(neighbors)
        if next_node not in subgraph:
            subgraph.append(next_node)

    return subgraph

def uni_features(graph, subgraph_list, mu_0=0., mu_1=5., std_0=1., std_1=1.):

    feature_list = []
    for t in range(len(subgraph_list)):
        subgraph = subgraph_list[t]
        features = []
        for node in graph.nodes():
            if node in subgraph:
                feature = np.random.normal(mu_1, std_1)
            else:
                feature = np.random.normal(mu_0, std_0)

            features.append(feature)

        feature_list.append(features)

    return np.array(feature_list)

def generate_subgraph_overlap_list(num_time_stamps, start_time_stamps, end_time_stamps, num_nodes_subgraph_min, num_nodes_subgraph_max, overlap_ratio):

    num_signal_time_stamps = end_time_stamps - start_time_stamps + 1
    if 1 == num_signal_time_stamps % 2:
        first_half_signal_interval = list(np.linspace(num_nodes_subgraph_min, num_nodes_subgraph_max, num=int(num_signal_time_stamps/2.)+1, endpoint=True, dtype=np.int))
        last_half_signal_interval = first_half_signal_interval[-2::-1]
    else:
        first_half_signal_interval = list(np.linspace(num_nodes_subgraph_min, num_nodes_subgraph_max,
                                                 num=int(num_signal_time_stamps / 2.), endpoint=True, dtype=np.int))
        last_half_signal_interval = first_half_signal_interval[-1::-1]

    signal_interval = first_half_signal_interval + last_half_signal_interval

    num_nodes_subgraph_list = []
    for t in range(num_time_stamps):
        if t < start_time_stamps:
            num_nodes_subgraph_list.append(0)
        elif t > end_time_stamps:
            num_nodes_subgraph_list.append(0)
        else:
            num_nodes_subgraph_list.append(signal_interval[t-start_time_stamps])

    num_nodes_overlap_list = []
    for t in range(1, num_time_stamps):
        num_nodes_prev_subgraph = num_nodes_subgraph_list[t-1]
        num_nodes_cur_subgraph = num_nodes_subgraph_list[t]
        num_nodes_overlap = int(overlap_ratio * np.min([num_nodes_prev_subgraph, num_nodes_cur_subgraph]))
        num_nodes_overlap_list.append(num_nodes_overlap)

    return num_nodes_subgraph_list, num_nodes_overlap_list


def generate_instance(num_nodes, num_time_stamps, num_nodes_subgraph_list, num_nodes_overlap_list, mu_0=0., mu_1=5., std_0=1., std_1=1.):

    graph = grid_graph_structure(num_nodes)

    subgraph_list = generate_subgraphs(graph, num_time_stamps, num_nodes_subgraph_list, num_nodes_overlap_list)

    features_array = uni_features(graph, subgraph_list, mu_0, mu_1, std_0, std_1)

    instance = {}
    instance['graph'] = graph
    instance['subgraphs'] = subgraph_list
    instance['features'] = features_array
    # logger.debug(subgraph_list)
    # logger.debug(features_array)

    return instance

def generate_dataset(wfn, num_instances, num_nodes, num_time_stamps, start_time_stamps, end_time_stamps, num_nodes_subgraph_min, num_nodes_subgraph_max, overlap_ratio, mu_0=0., mu_1=5., std_0=1., std_1=1.):

    dataset = []
    start_time = time.time()
    logger.debug('-' * 5 + ' start generating ' + '-' * 5)
    num_nodes_subgraph_list, num_nodes_overlap_list = generate_subgraph_overlap_list(num_time_stamps, start_time_stamps, end_time_stamps, num_nodes_subgraph_min, num_nodes_subgraph_max, overlap_ratio)
    for i in range(num_instances):
        logger.debug('instance: {:d}'.format(i+1))
        instance = generate_instance(num_nodes, num_time_stamps, num_nodes_subgraph_list, num_nodes_overlap_list, mu_0=mu_0, mu_1=mu_1, std_0=std_0, std_1=std_1)
        dataset.append(instance)

    logger.debug('-' * 5 + ' start dumping dataset to a file ' + '-' * 5)
    with open(wfn, 'wb') as wfile:
        pickle.dump(dataset, wfile)

    logger.debug('finished, run time: {:.5f}'.format(time.time() - start_time))


if __name__ == '__main__':
    num_instances = 1
    num_nodes = 100
    num_time_stamps = 9
    num_time_stamps_signal = 5
    start_time_stamps = num_time_stamps / 2 - num_time_stamps_signal / 2
    end_time_stamps = num_time_stamps / 2 + num_time_stamps_signal / 2
    num_nodes_subgraph_min = 10
    num_nodes_subgraph_max = 20
    overlap_ratio = 0.8
    mu_0 = 0.
    mu_1 = 5.

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/synthetic'
    fn = 'syn_mu_{:.1f}_{:.1f}_num_{:d}_{:d}_{:d}_{:d}_time_{:d}_{:d}_{:d}.pkl'.format(mu_0, mu_1, num_instances, num_nodes, num_nodes_subgraph_min, num_nodes_subgraph_max, start_time_stamps, end_time_stamps, num_time_stamps)
    wfn = os.path.join(path, fn)
    generate_dataset(wfn, num_instances, num_nodes, num_time_stamps, start_time_stamps, end_time_stamps, num_nodes_subgraph_min, num_nodes_subgraph_max, overlap_ratio, mu_0=0., mu_1=5.)