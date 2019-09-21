#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : performance
# @Date     : 07/21/2019 13:50:55
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :

from __future__ import print_function

import os
import pickle

import numpy as np
import networkx as nx


def calculate_performance(id):
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions/exp_1/case_{}/Output_{}'.format(id, id)
    fn = '01DenseSubgraphNodes.txt'
    subgraph = set()
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            node = int(line.strip())
            subgraph.add(node)

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions/exp_1/case_{}'.format(id)
    fn = 'attributes_biased_{}.pkl'.format(id)
    with open(os.path.join(path, fn), 'rb') as rfile:
        dataset = pickle.load(rfile)

    true_subgraph = dataset['subgraph']
    prec, rec, fm, iou = evaluate(true_subgraph, subgraph)

    print(prec, rec, fm, iou)


def calculate_perfomance_2(id):
    print('noseed')
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic/dcs/case_{}/Output'.format(id)
    # path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic/dcs/case_{}/Output_top3'.format(id)
    fn = '01DenseSubgraphNodes.txt'
    subgraph = set()
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            node = int(line.strip())
            subgraph.add(node)

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic'
    fn = 'dual_nodes_1000_deg1_3_deg2_10_subsize_100_{}.pkl'.format(id)
    with open(os.path.join(path, fn), 'rb') as rfile:
        instance = pickle.load(rfile)

    true_subgraph = instance['true_subgraph']
    first_graph = instance['first_graph']
    second_graph = instance['second_graph']
    prec, rec, fm, iou = evaluate(true_subgraph, subgraph)
    print(prec, rec, fm, iou, sep='\t', end='\t')
    print(nx.density(first_graph), end='\t')
    print(nx.density(nx.subgraph(first_graph, subgraph)), end='\t')
    print(nx.density(second_graph), end='\t')
    print(nx.density(nx.subgraph(second_graph, subgraph)), end='\t')

    # print('top1')
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic/dcs/case_{}/Output_top1'.format(id)
    fn = '01DenseSubgraphNodes.txt'
    subgraph = set()
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            node = int(line.strip())
            subgraph.add(node)

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic'
    fn = 'dual_nodes_1000_deg1_3_deg2_10_subsize_100_{}.pkl'.format(id)
    with open(os.path.join(path, fn), 'rb') as rfile:
        instance = pickle.load(rfile)

    true_subgraph = instance['true_subgraph']
    first_graph = instance['first_graph']
    second_graph = instance['second_graph']
    prec, rec, fm, iou = evaluate(true_subgraph, subgraph)
    print(prec, rec, fm, iou, sep='\t', end='\t')
    print(nx.density(first_graph), end='\t')
    print(nx.density(nx.subgraph(first_graph, subgraph)), end='\t')
    print(nx.density(second_graph), end='\t')
    print(nx.density(nx.subgraph(second_graph, subgraph)), end='\t')

    # print('top2')
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic/dcs/case_{}/Output_top2'.format(id)
    fn = '01DenseSubgraphNodes.txt'
    subgraph = set()
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            node = int(line.strip())
            subgraph.add(node)

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic'
    fn = 'dual_nodes_1000_deg1_3_deg2_10_subsize_100_{}.pkl'.format(id)
    with open(os.path.join(path, fn), 'rb') as rfile:
        instance = pickle.load(rfile)

    true_subgraph = instance['true_subgraph']
    first_graph = instance['first_graph']
    second_graph = instance['second_graph']
    prec, rec, fm, iou = evaluate(true_subgraph, subgraph)
    print(prec, rec, fm, iou, sep='\t', end='\t')
    print(nx.density(first_graph), end='\t')
    print(nx.density(nx.subgraph(first_graph, subgraph)), end='\t')
    print(nx.density(second_graph), end='\t')
    print(nx.density(nx.subgraph(second_graph, subgraph)), end='\t')

    # print('top3')
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic/dcs/case_{}/Output_top3'.format(id)
    fn = '01DenseSubgraphNodes.txt'
    subgraph = set()
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            node = int(line.strip())
            subgraph.add(node)

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic'
    fn = 'dual_nodes_1000_deg1_3_deg2_10_subsize_100_{}.pkl'.format(id)
    with open(os.path.join(path, fn), 'rb') as rfile:
        instance = pickle.load(rfile)

    true_subgraph = instance['true_subgraph']
    first_graph = instance['first_graph']
    second_graph = instance['second_graph']
    prec, rec, fm, iou = evaluate(true_subgraph, subgraph)
    print(prec, rec, fm, iou, sep='\t', end='\t')
    print(nx.density(first_graph), end='\t')
    print(nx.density(nx.subgraph(first_graph, subgraph)), end='\t')
    print(nx.density(second_graph), end='\t')
    print(nx.density(nx.subgraph(second_graph, subgraph)), end='\t')


def calculate_perfomance_3(id):
    # path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic/dcs/top_remove/case_{}/Output'.format(id)
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic/dcs/top_remove_2/case_{}/Output'.format(id)
    fn = '01DenseSubgraphNodes.txt'
    subgraph = set()
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            node = int(line.strip())
            subgraph.add(node)

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic'
    fn = 'dual_nodes_1000_deg1_3_deg2_10_subsize_100_{}.pkl'.format(id)
    with open(os.path.join(path, fn), 'rb') as rfile:
        instance = pickle.load(rfile)

    true_subgraph = instance['true_subgraph']
    first_graph = instance['first_graph']
    second_graph = instance['second_graph']
    prec, rec, fm, iou = evaluate(true_subgraph, subgraph)
    print(prec, rec, fm, iou, sep='\t', end='\t')
    print(nx.density(first_graph), end='\t')
    print(nx.density(nx.subgraph(first_graph, subgraph)), end='\t')
    print(nx.density(second_graph), end='\t')
    print(nx.density(nx.subgraph(second_graph, subgraph)), end='\t')


def calculate_performance_4(id):
    # path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Mus_Multiplex_Genetic/dcs/top_not_remove/case_{}/Output_top1'.format(id)
    # path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Mus_Multiplex_Genetic/dcs/top_remove/case_{}/Output'.format(id)
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Mus_Multiplex_Genetic/dcs/top_remove_2/case_{}/Output'.format(id)

    fn = '01DenseSubgraphNodes.txt'
    subgraph = set()
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            node = int(line.strip())
            subgraph.add(node)

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Mus_Multiplex_Genetic/dual'
    fn = 'dual_mus_case_{}.pkl'.format(id)
    with open(os.path.join(path, fn), 'rb') as rfile:
        instance = pickle.load(rfile)

    true_subgraph = instance['true_subgraph']
    first_graph = instance['first_graph']
    second_graph = instance['second_graph']
    prec, rec, fm, iou = evaluate(true_subgraph, subgraph)
    print(prec, rec, fm, iou, sep='\t', end='\t')
    print(nx.density(first_graph), end='\t')
    print(nx.density(nx.subgraph(first_graph, subgraph)), end='\t')
    print(nx.density(second_graph), end='\t')
    print(nx.density(nx.subgraph(second_graph, subgraph)), end='\t')


def evaluate(true_subgraph, pred_subgraph):
    true_subgraph, pred_subgraph = set(true_subgraph), set(pred_subgraph)

    prec = rec = fm = iou = 0.

    intersection = true_subgraph & pred_subgraph
    union = true_subgraph | pred_subgraph

    if not 0. == len(pred_subgraph):
        prec = len(intersection) / float(len(pred_subgraph))

    if not 0. == len(true_subgraph):
        rec = len(intersection) / float(len(true_subgraph))

    if prec + rec > 0.:
        fm = (2. * prec * rec) / (prec + rec)

    if union:
        iou = len(intersection) / float(len(union))

    return prec, rec, fm, iou


def check_dblp_result():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/DBLP/DBLP_Citation_2014_May/DM'
    # path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/DBLP/DBLP_Citation_2014_May/DB'
    fn = 'dm_top30000_dataset.pkl'
    # fn = 'db_top30000_dataset.pkl'

    with open(os.path.join(path, fn), 'rb') as rfile:
        dataset = pickle.load(rfile)

    first_graph = dataset['first_graph']
    second_graph = dataset['second_graph']

    print('statistics')
    print(first_graph.number_of_nodes())
    print(first_graph.number_of_edges())
    print(second_graph.number_of_nodes())
    print(second_graph.number_of_edges())

    # lcc = max(nx.connected_component_subgraphs(second_graph), key=len)
    # subgraph = nx.subgraph(second_graph, lcc)
    # print(nx.is_connected(subgraph))
    # print(subgraph.number_of_nodes())
    # print(subgraph.number_of_edges())
    #
    # first_graph = dataset['first_graph']
    # subgraph = nx.subgraph(first_graph, lcc)
    # print(nx.is_connected(subgraph))
    # print(subgraph.number_of_nodes())
    # print(subgraph.number_of_edges())

    density = nx.density(first_graph)
    print('density of first graph: {:.5f}'.format(density))
    density = nx.density(second_graph)
    print('density of second graph: {:.5f}'.format(density))

    # print(sorted(list(subgraph)))
    # print(len(subgraph))

    original_subgraph = [86, 96, 247, 399, 400, 508, 513, 578, 579, 602, 672, 753, 789, 801, 925, 992, 1135, 1406, 1423, 1440, 1574, 1632, 1636, 1769, 1822, 1836, 1908, 1912, 1951, 2040, 2072, 2202, 2377, 2392, 2398, 2408, 2442, 2447, 2626, 2711, 2839, 2871, 2910, 2960, 2992, 3036, 3162, 3228, 3246, 3259, 3308, 3385, 3393, 3614, 3650, 3764, 3771, 3843, 3850, 3991, 3996, 4094] # DM ght
    # original_subgraph = [109, 123, 143, 155, 191, 209, 249, 335, 491, 510, 523, 529, 534, 600, 653, 678, 708, 739, 830, 846, 857, 938, 961, 981, 1057, 1110, 1183, 1219, 1237, 1330, 1334, 1370, 1390, 1448, 1499, 1512, 1570, 1590, 1625, 1626, 1628, 1645, 1663, 1685, 1754, 1758, 1761, 1878, 1903, 1908, 1921, 1940, 1967, 2015, 2042, 2067, 2111, 2139, 2140, 2171, 2239, 2240, 2247, 2284, 2364, 2365, 2375, 2382, 2386, 2463, 2511, 2514, 2574, 2588, 2608, 2630, 2709, 2729, 2749, 2753, 2929, 3061, 3069, 3113, 3141, 3191, 3221, 3259, 3281, 3291, 3295, 3317, 3319, 3333, 3415, 3437, 3469, 3487, 3586, 3602, 3612, 3628, 3637, 3830, 3848, 3879, 3890, 3943, 3949, 3973, 4027, 4035, 4045, 4047, 4064, 4072, 4079, 4096, 4142, 4168, 4193, 4199, 4234, 4255, 4265, 4284, 4292, 4345, 4356, 4368, 4371, 4395] # DB ghtp

    # original_subgraph = []
    #     # dm_path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/DBLP/DBLP_Citation_2014_May/dcs_dm/Output'
    #     # # db_path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/DBLP/DBLP_Citation_2014_May/dcs_db/Output'
    #     # fn = '01DenseSubgraphNodes.txt'
    #     # with open(os.path.join(dm_path, fn)) as rfile:
    #     # # with open(os.path.join(db_path, fn)) as rfile:
    #     #     for line in rfile:
    #     #         term = int(line.strip())
    #     #         original_subgraph.append(term)
    #     #
    #     # print(original_subgraph)

    fn = 'dm_author_map.pkl'
    # fn = 'db_author_map.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        dm_author_map = pickle.load(rfile)

    new_old = dm_author_map['new_old']
    old_new = dm_author_map['old_new']

    # print(dm_author_map.keys())

    fn = 'dm_authors.pkl'
    # fn = 'db_authors.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        dm_author = pickle.load(rfile)

    fn = 'dm_author_paper_count.pkl'
    # fn = 'db_author_paper_count.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        dm_author_paper_count = pickle.load(rfile)
    # print(dm_author_paper_count[0])
    print(len(dm_author_paper_count))
    print(dm_author_paper_count)


    subgraph = nx.subgraph(first_graph, original_subgraph)
    print(subgraph.number_of_nodes())
    print(subgraph.number_of_edges())
    print(nx.is_connected(subgraph))
    lcc = max(nx.connected_component_subgraphs(subgraph), key=len)
    print(len(lcc))

    for node in subgraph.nodes():
        for name in dm_author.keys():
            if dm_author[name] == new_old[node]:
                print(name, nx.degree(subgraph, node), sep=',')

    for edge in subgraph.edges():
        node_1, node_2 = edge
        # print(node_1, node_2, new_old[node_1], new_old[node_2])
        for name in dm_author.keys():
            if dm_author[name] == new_old[node_1]:
                name_1 = name
    #
            if dm_author[name] == new_old[node_2]:
                name_2 = name
    #
        # print(node_1, node_2, new_old[node_1], new_old[node_2], name_1, name_2)
        print('{},{}'.format(name_1, name_2))

    subgraph = nx.subgraph(second_graph, original_subgraph)
    # subgraph = nx.subgraph(second_graph, lcc)
    print(subgraph.number_of_nodes())
    print(subgraph.number_of_edges())
    print(nx.is_connected(subgraph))
    density = nx.density(nx.subgraph(second_graph, subgraph))
    print('density of subgraph in second graph: {}'.format(density))

    # lcc = max(nx.connected_component_subgraphs(subgraph), key=len)
    # print(len(lcc))

    # subgraph = nx.subgraph(second_graph, lcc)


    x = 0.
    z = []
    for node in subgraph.nodes():
        for name in dm_author.keys():
            if dm_author[name] == new_old[node]:
                # print(node, new_old[node], name, dm_author_paper_count[node])
                # print(name, dm_author_paper_count[node])
                print(name, nx.degree(subgraph, node))
                x += dm_author_paper_count[node]
                z.append(dm_author_paper_count[node])

    print(len(z), z)

    print('average paper count: {:.5f}'.format(x / len(subgraph)))

    for edge in subgraph.edges():
        node_1, node_2 = edge
        # print(node_1, node_2, new_old[node_1], new_old[node_2])
        for name in dm_author.keys():
            if dm_author[name] == new_old[node_1]:
                name_1 = name
    #
            if dm_author[name] == new_old[node_2]:
                name_2 = name
    #
        # print(node_1, node_2, new_old[node_1], new_old[node_2], name_1, name_2)
        print('{},{}'.format(name_1, name_2))


if __name__ == '__main__':

    # for id in range(10):
    # calculate_performance(id)

    # calculate_perfomance_2()

    # calculate_perfomance_3(9)

    # calculate_performance_4(9)

    check_dblp_result()