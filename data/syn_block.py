#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : syn_block
# @Date     : 04/08/2019 22:55:06
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

import nxmetis
import numpy as np
import networkx as nx


def ba_graph_structure(num_nodes, prob=0.1, m=None, seed=None):
    graph = nx.barabasi_albert_graph(num_nodes, m, seed)

    logger.debug('-' * 5 + ' network info ' + '-' * 5)
    logger.debug('-' * 5 + 'network info' + '-' * 5)
    logger.debug('number of nodes: {:d}'.format(nx.number_of_nodes(graph)))
    logger.debug('number of edges: {:d}'.format(nx.number_of_edges(graph)))
    logger.debug("density: {:.5f}".format(nx.density(graph)))
    all_deg = []
    for idx, deg in nx.degree(graph):
        all_deg.append(deg)
    logger.debug("average degree: {:.5f}".format(np.mean(all_deg)))
    logger.debug('number of connected components in the graph: {:d}'.format(nx.number_connected_components(graph)))
    largest_component = max(nx.connected_component_subgraphs(graph), key=len)
    logger.debug('number of nodes in the largest component: {:d}'.format(largest_component.number_of_nodes()))

    return graph

def generate_subgraph(graph, num_nodes_subgraph):
    subgraph = []
    num_nodes = graph.number_of_nodes()
    current_node = np.random.choice(num_nodes)
    while len(subgraph) < num_nodes_subgraph:
        neighbors = list(nx.neighbors(graph, current_node))
        next_node = np.random.choice(neighbors)
        if next_node not in subgraph:
            subgraph.append(next_node)

        current_node = next_node

    return sorted(subgraph)


def uni_features(graph, subgraph, mu_0=0., mu_1=5., std_0=1., std_1=1.):
    features = []
    for node in graph.nodes():
        if node in subgraph:
            feature = np.random.normal(mu_1, std_1)
        else:
            feature = np.random.normal(mu_0, std_0)

        features.append(feature)

    return np.array(features)


def patition(graph, num_blocks):

    block_node_sets = []
    node_block_dict = {}
    block_id = -1
    logger.debug('-' * 5 + ' starting partitioning ' + '-' * 5)
    (edge_cut, partitions) = nxmetis.partition(graph, num_blocks)
    logger.debug('number of edge cuts: {:d}'.format(edge_cut))

    for ind, partition in enumerate(partitions):
        subgraph = nx.subgraph(graph, partition)
        logger.debug('partion: {:d}'.format(ind))
        logger.debug('number of nodes: {:d}'.format(subgraph.number_of_nodes()))
        logger.debug('number of edges: {:d}'.format(subgraph.number_of_edges()))
        logger.debug('number of connected components: {:d}'.format(nx.number_connected_components(subgraph)))
        logger.debug('-' * 5)

        block_node_sets.append(sorted(partition))
        for node in subgraph.nodes:
            node_block_dict[node] = ind

    block_boundary_edges_dict = {} # key - block id, value - boundary edge list
    cut = 0
    for edge in graph.edges():
        node_1, node_2 = edge
        node_1_block_id = node_block_dict[node_1]
        node_2_block_id = node_block_dict[node_2]
        if not node_1_block_id == node_2_block_id:
            cut += 1
            if node_1_block_id not in block_boundary_edges_dict:
                block_boundary_edges_dict[node_1_block_id] = []
            if node_2_block_id not in block_boundary_edges_dict:
                block_boundary_edges_dict[node_2_block_id] = []
            block_boundary_edges_dict[node_1_block_id].append((node_1, node_2))
            block_boundary_edges_dict[node_2_block_id].append((node_2, node_1))

    logger.debug('number of edge cuts: {:d}'.format(cut))


    return block_node_sets, node_block_dict, block_boundary_edges_dict


def generate_ba_instance(num_nodes, num_nodes_subgraph, num_blocks, m, mu_0, mu_1, std_0=1., std_1=1.):
    instance = {}

    graph = ba_graph_structure(num_nodes, m=m)
    subgraph = generate_subgraph(graph, num_nodes_subgraph)

    block_node_sets, node_block_dict, block_boundary_edges_dict = patition(graph, num_blocks)

    features = uni_features(graph, subgraph, mu_0, mu_1, std_0, std_1)

    instance['graph'] = graph
    instance['subgraph'] = subgraph
    instance['features'] = features
    instance['block_node_sets'] = block_node_sets
    instance['node_block_dict'] = node_block_dict
    instance['block_boundary_edges_dict'] = block_boundary_edges_dict

    return instance


def generate_dataset(wfn, num_instances, num_nodes, num_nodes_subgraph, num_blocks, m, mu_0, mu_1, std_0=1., std_1=1.):

    dataset = []
    start_time = time.time()
    logger.debug('-' * 5 + ' start generating ' + '-' * 5)
    for i in range(num_instances):
        logger.debug('instance: {:d}'.format(i+1))
        instance = generate_ba_instance(num_nodes, num_nodes_subgraph, num_blocks, m, mu_0, mu_1, std_0, std_1)
        dataset.append(instance)

    logger.debug('-' * 5 + ' start dumping dataset to a file ' + '-' * 5)
    with open(wfn, 'wb') as wfile:
        pickle.dump(dataset, wfile)

    logger.debug('finished, run time: {:.5f}'.format(time.time() - start_time))




if __name__ == '__main__':

    num_instances = 1
    num_nodes = 1000
    num_nodes_subgraph = 100
    num_blocks = 5
    m = 3
    mu_0 = 0.
    mu_1 = 5.

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/synthetic'
    fn = 'syn_mu_{:.1f}_{:.1f}_num_{:d}_{:d}_{:d}.pkl'.format(mu_0, mu_1, num_instances, num_nodes, num_nodes_subgraph, num_blocks)
    wfn = os.path.join(path, fn)
    generate_dataset(wfn, num_instances, num_nodes, num_nodes_subgraph, num_blocks, m, mu_0, mu_1)