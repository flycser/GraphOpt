#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : generate_ba_graph
# @Date     : 08/11/2019 10:54:57
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :



from __future__ import print_function


import os
import pickle

import numpy as np
import networkx as nx


def generate_dataset():

    dataset = []
    mu = 3

    for case_id in range(10):
        fn = '/network/rit/lab/ceashpc/share_data/GraphOpt/DCS/dense/syn/graph_{}.pkl'.format(case_id)
        with open(fn, 'rb') as rfile:
            graph = pickle.load(rfile)

        fn = '/network/rit/lab/ceashpc/share_data/GraphOpt/DCS/dense/syn/edges_{}.log'.format(case_id)
        with open(fn) as rfile:
            while True:
                line = rfile.readline()
                if 'Subgraph' in line:
                    line = rfile.readline().strip()
                    subgraph = [int(term) for term in line[1:-1].split()]
                    print(subgraph)

                    break

        attributes = np.zeros(graph.number_of_nodes())
        for node in graph.nodes():
            if node in subgraph:
                attributes[node] = np.random.normal(mu, 1)
            else:
                attributes[node] = np.random.normal(0, 1)

        print([attribute for attribute in attributes])

        dataset.append({'graph': graph, 'weight': attributes, 'true_subgraph': subgraph})

    fn = '/network/rit/lab/ceashpc/share_data/GraphOpt/DCS/dense/syn/dataset_mu{}.pkl'.format(mu)
    with open(fn, 'wb') as wfile:
        pickle.dump(dataset, wfile)


def generate_dataset_2():
    dataset = []
    elevate_val = 0.5

    for case_id in range(10):
        fn = '/network/rit/lab/ceashpc/share_data/GraphOpt/DCS/dense/syn/graph_{}.pkl'.format(case_id)
        with open(fn, 'rb') as rfile:
            graph = pickle.load(rfile)

        fn = '/network/rit/lab/ceashpc/share_data/GraphOpt/DCS/dense/syn/edges_{}.log'.format(case_id)
        with open(fn) as rfile:
            while True:
                line = rfile.readline()
                if 'Subgraph' in line:
                    line = rfile.readline().strip()
                    subgraph = [int(term) for term in line[1:-1].split()]
                    print(sorted(subgraph))

                    break

        attributes = np.zeros(graph.number_of_nodes())
        # uniform distribution, elevate value
        for node in graph.nodes():
            if node in subgraph:
                attributes[node] = np.random.uniform(0., 1.) + elevate_val # subgraph nodes
            else: # background nodes, uniform distribution [0, 1)
                attributes[node] = np.random.uniform(0., 1.)

        print([attribute for attribute in attributes])

        dataset.append({'graph': graph, 'weight': attributes, 'true_subgraph': subgraph})

    fn = '/network/rit/lab/ceashpc/share_data/GraphOpt/DCS/dense/syn/dataset_ele{}.pkl'.format(elevate_val)
    with open(fn, 'wb') as wfile:
        pickle.dump(dataset, wfile)



def extract_subgraph():
    fn = '/network/rit/lab/ceashpc/share_data/GraphOpt/DCS/dense/syn/edges_7.log'
    with open(fn) as rfile:
        while True:
            line = rfile.readline()
            if 'Subgraph' in line:
                line = rfile.readline().strip()
                subgraph = [int(term) for term in line[1:-1].split()]
                print(sorted(subgraph))

                break

def test():
    fn = '/network/rit/lab/ceashpc/share_data/GraphOpt/DCS/dense/Peel-C++/tmp_2.txt'
    count = 0
    graph = nx.Graph()
    with open(fn) as rfile:
        for line in rfile:
            if count == 0:
                count += 1
                continue
            else:
                terms = line.strip().split('\t')
                node_1, node_2 = int(terms[0]), int(terms[1])

                graph.add_edge(node_1, node_2)

    print(graph.number_of_nodes())
    print(graph.number_of_edges())

    subgraph = '23 398 34 350 36 594 41 194 56 901 57 420 64 744 87 303 92 672 106 111 391 120 122     736 130 142 353 149 807 154 460 155 701 157 898 172 419 174 177 178 204 196 533 213 516 220 222 226 931 229 372 240 277 257 650 266 268 466 278 752 281 308 309 966 311 714 316 329 25 385 77 332 939 244 342 358 389 363 727 369 452 415 421 491 426 967 429 463 562 469 471 676 485 578 489 952 503 547 531 549 618 914 625 627 628 210 276 635 668 987 682 703 711 948 275 880 732 746 757 758 763 786 789 816 824 878 913 919 921 924 925 974'
    subgraph = [int(node) for node in subgraph.split()]
    print(nx.density(graph))
    print(nx.density(nx.subgraph(graph, subgraph)))


if __name__ == '__main__':

    # generate_dataset()

    generate_dataset_2()

    # extract_subgraph()

    # for case_id in range(10):
    #     while True:
    #         n = 1000
    #         p = 0.007
    #         graph = nx.erdos_renyi_graph(n, p)
    #
    #         print(graph.number_of_nodes())
    #         print(graph.number_of_edges())
    #
    #         print(nx.is_connected(graph))
    #         if nx.is_connected(graph):
    #             fn = '/network/rit/lab/ceashpc/share_data/GraphOpt/DCS/dense/syn/edges_{}.txt'.format(case_id)
    #             with open(fn, 'w') as wfile:
    #                 wfile.write('{}\t{}\n'.format(graph.number_of_nodes(), graph.number_of_edges()))
    #                 for edge in graph.edges():
    #                     wfile.write('{}\t{}\n'.format(edge[0], edge[1]))
    #                 # print('{}\t{}'.format(edge[0], edge[1]))
    #                 # print('{},{}'.format(edge[0], edge[1]))
    #             fn = '/network/rit/lab/ceashpc/share_data/GraphOpt/DCS/dense/syn/graph_{}.pkl'.format(case_id)
    #             with open(fn, 'wb') as wfile:
    #                 pickle.dump(graph, wfile)
    #
    #             break

    # test()


    # fn = 'edges_3.txt'
    # with open(fn) as rfile:
    #     graph = nx.read_edgelist(rfile, delimiter=' ', nodetype=int)
    #
    # print(graph.number_of_nodes())
    # print(graph.number_of_edges())