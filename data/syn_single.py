#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : generate
# @Date     : 03/27/2019 22:36:49
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

from utils import subgraph_random_walk, visual_grid_graph


def grid_graph_structure(num_nodes):

    width = height = int(np.sqrt(num_nodes))
    graph = nx.grid_graph(dim=[width, height])

    # change node indices
    mapping = {}
    for (i, j) in graph .nodes():
        mapping[(i, j)] = i * width + j
    graph  = nx.relabel_nodes(graph , mapping)
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

def er_graph_structure(num_nodes, prob=0.1, seed=None):

    graph = nx.fast_gnp_random_graph(num_nodes, prob, seed)
    logger.debug('-' * 5 + 'network info' + '-' * 5)
    logger.debug('number of nodes: {:d}'.format(nx.number_of_nodes(graph)))
    logger.debug('number of edges: {:d}'.format(nx.number_of_edges(graph)))
    all_deg = []
    for idx, deg in nx.degree(graph):
        all_deg.append(deg)
    logger.debug('average degree: {:d}'.format(np.mean(all_deg)))
    logger.debug('connected components in the graph: {:d}'.format(nx.number_connected_components(graph)))
    largest_component = max(nx.connected_component_subgraphs(graph), key=len)
    logger.debug('the largest component size: {:d}'.format(largest_component.number_of_nodes()))

    return graph

def ba_graph_structure(num_nodes, m=None, seed=None):
    
    graph = nx.barabasi_albert_graph(num_nodes, m, seed)
    logger.debug('-' * 5 + ' network info ' + '-' * 5)
    logger.debug('number of nodes: {:d}'.format(nx.number_of_nodes(graph)))
    logger.debug('number of edges: {:d}'.format(nx.number_of_edges(graph)))
    all_deg = []
    for idx, deg in nx.degree(graph):
        all_deg.append(deg)
    logger.debug('average degree: {:d}'.format(np.mean(all_deg)))
    logger.debug('connected components in the graph: {:d}'.format(nx.number_connected_components(graph)))
    largest_component = max(nx.connected_component_subgraphs(graph), key=len)
    logger.debug('the largest component size: {:d}'.format(largest_component.number_of_nodes()))

    return graph

def uni_features(graph, subgraph, mu_0=0., mu_1=5., std_0=1., std_1=1.):
    features = np.zeros(graph.number_of_nodes())
    for node in graph.nodes():
        if node in subgraph:
            features[node] = np.random.normal(mu_1, std_1)
        else:
            features[node] = np.random.normal(mu_0, std_0)

    return features

def generate_instance(num_nodes, num_nodes_subgraph, mu_0=0., mu_1=5., std_0=1., std_1=1.):

    # generate grid structure
    graph = grid_graph_structure(num_nodes)
    # generate subgraph
    subgraph = subgraph_random_walk(graph, num_nodes_subgraph)
    # generate univariate features
    features = uni_features(graph, subgraph, mu_0, mu_1, std_0, std_1)

    # n - normal, anl - anomalous
    return {'graph':graph, 'subgraph':subgraph, 'features':features, 'mu_n':mu_0, 'mu_anl':mu_1, 'std_n':std_0, 'std_anl':std_1}


def generate_dataset(wfn, num_instances, num_nodes, num_nodes_subgraph, mu_0=0., mu_1=5., std_0=1., std_1=1.):
    """
    dump multiple instances to a file
    :param wfn:
    :return:
    """

    dataset = []
    start_time = time.time()
    logger.debug('-' * 5 + ' start generating ' + '-' * 5)
    for i in range(num_instances):
        logger.debug('instance: {:d}'.format(i+1))
        instance = generate_instance(num_nodes, num_nodes_subgraph, mu_0, mu_1, std_0, std_1)
        dataset.append(instance)

    logger.debug('-' * 5 + ' start dumping dataset to a file ' + '-' * 5)
    with open(wfn, 'wb') as wfile:
        pickle.dump(dataset, wfile)

    logger.debug('finished, run time: {:.5f}'.format(time.time() - start_time))


if __name__ == '__main__':

    num_instances = 1
    num_nodes = 100
    num_subgraph_nodes = 10
    mu_0 = 0.
    mu_1 = 5.


    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/synthetic'
    fn = 'syn_mu_{:.1f}_{:.1f}_num_{:d}_{:d}_{:d}.pkl'.format(mu_0, mu_1, num_instances, num_nodes, num_subgraph_nodes)
    wfn = os.path.join(path, fn)
    generate_dataset(wfn, num_instances, num_nodes, num_subgraph_nodes, mu_0, mu_1)

    rfn = os.path.join(path, fn)
    with open(rfn, 'rb') as rfile:
        dataset = pickle.load(rfile)

    visual_grid_graph(10, 10, dataset[0]['subgraph'])