#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : generate_dual_networks_2
# @Date     : 09/18/2019 12:03:29
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


from __future__ import print_function

import os
import sys
import pickle

import numpy as np
import networkx as nx


def generate_dcs_dataset(id):
    mu = 3
    case_id = id
    restart = 0.3
    rpath = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Homo_Multiplex_Genetic/dual'
    wpath = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Homo_Multiplex_Genetic/dcs/mu{}/top_not_remove/case_{}'.format(mu, case_id)

    fn = 'dual_mu_{}_case_{}.pkl'.format(mu, case_id)
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

    print('true subgraph', sorted(list(instance['true_subgraph'])))


if __name__ == '__main__':

    case_id = 0

    generate_dcs_dataset(case_id)

