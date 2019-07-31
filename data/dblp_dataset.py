#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : dblp_dataset
# @Date     : 06/01/2019 21:07:12
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


from __future__ import print_function

import os
import re
import pickle
import random
import datetime

import nxmetis
import numpy as np
import networkx as nx

import apdm


def extract():

    path = '/network/rit/lab/ceashpc/share_data/DBLP/dblp_coauthor'
    fn = 'out.dblp_coauthor'
    # earliest_timestamp = 631152000 # 1990
    # earliest_timestamp = 788918400 # 1995
    earliest_timestamp = 946684800 # 2000
    # earliest_timestamp = 1104537600 # 2005
    # earliest_timestamp = 1262304000 # 2010
    # latest_timestamp = 1388534461 # 2013
    # latest_timestamp = 946684800 # 2000
    latest_timestamp = 1104537600 # 2005
    # latest_timestamp = 1262304000 # 2010

    unixts_year_map = {631152000:1990, 788918400:1995, 946684800:2000, 1104537600:2005, 1262304000:2010, 1388534461:2013}

    with open(os.path.join(path, fn)) as rfile, open(os.path.join(path, 'out.{}-{}.dblp_coauthor'.format(unixts_year_map[earliest_timestamp], unixts_year_map[latest_timestamp])), 'w') as wfile:
        count = 0
        for line in rfile:
            if not line[0].isdigit():
                continue
            line = line.strip()
            terms = re.split('\s+', line)
            if len(terms) < 4:
                continue

            print(count)
            count += 1
            # author1 = terms[0]
            # author2 = terms[1]
            # weight = int(terms[2])
            timestamp = int(terms[3])
            if earliest_timestamp <= timestamp <= latest_timestamp:
                wfile.write(line + '\n')

def construct():
    path = '/network/rit/lab/ceashpc/share_data/DBLP/dblp_coauthor'
    start_year = 2005
    end_year = 2010
    fn = 'out.{}-{}.dblp_coauthor'.format(start_year, end_year)

    graph = nx.Graph()
    records = {}
    with open(os.path.join(path, fn)) as rfile:
        count = 0
        for line in rfile:
            line = line.strip()
            terms = re.split('\s+', line)
            print(count)
            count += 1
            author1 = int(terms[0])
            author2 = int(terms[1])
            # weight = int(terms[2])
            timestamp = int(terms[3])
            if author1 not in records:
                records[author1] = {}
                records[author1][timestamp] = [author2]
            elif timestamp not in records[author1]:
                records[author1][timestamp] = [author2]
            else:
                records[author1][timestamp].append(author2)

            if author2 not in records:
                records[author2] = {}
                records[author2][timestamp] = [author1]
            elif timestamp not in records[author2]:
                records[author2][timestamp] = [author1]
            else:
                records[author2][timestamp].append(author1)

            graph.add_edge(author1, author2)

    dataset = {}
    dataset['graph'] = graph
    dataset['records'] = records

    lcc = max(nx.connected_component_subgraphs(graph), key=len)
    print('{}-{}'.format(start_year, end_year))
    print(graph.number_of_nodes())
    print(graph.number_of_edges())
    print(len(lcc))

    fn = 'dblp_{}-{}.pkl'.format(start_year, end_year)
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(dataset, wfile)

def generate_graph_feature():
    path = '/network/rit/lab/ceashpc/share_data/DBLP/dblp_coauthor'
    start_year = 1995
    end_year = 2005
    fn = 'dblp_{}-{}.pkl'.format(start_year, end_year)
    with open(os.path.join(path, fn), 'rb') as rfile:
        dataset = pickle.load(rfile)

    graph = dataset['graph']
    records = dataset['records']

    features = []
    timestamps = set()
    for author in records:
        # print(author, records[author])
        for key in records[author]:
            timestamps.add(key)
        features.append(len(records[author]))

    print(max(features))
    print(min(features))
    print(len(timestamps))

    # print(graph.number_of_nodes())
    # print(graph.number_of_edges())
    # print(nx.is_connected(graph))
    #
    # # print(records[228059])
    #
    # lcc = max(nx.connected_component_subgraphs(graph), key=len)
    # # print(type(lcc))
    # print(lcc.number_of_nodes())
    # print(lcc.number_of_edges())
    #
    # sorted_nodes = sorted([node for node in lcc.nodes()])
    # node_idx_map = {}
    # for idx, node in enumerate(sorted_nodes):
    #     node_idx_map[node] = idx
    #
    # connected_graph = nx.relabel_nodes(lcc, node_idx_map)
    # features = np.zeros(connected_graph.number_of_nodes())
    # for node in lcc:
    #     feature = len(records[node])
    #     features[node_idx_map[node]] = feature
    #
    # dataset = {}
    # dataset['graph'] = connected_graph
    # dataset['features'] = features
    # print(features.max(), features.min())
    #
    # fn = 'graph_feature_{}-{}.pkl'.format(start_year, end_year)
    # with open(os.path.join(path, fn), 'wb') as wfile:
    #     pickle.dump(dataset, wfile)


def generate_dataset():
    path = '/network/rit/lab/ceashpc/share_data/DBLP/dblp_coauthor'
    start_year = 1995
    end_year = 2005
    fn = 'graph_feature_{}-{}.pkl'.format(start_year, end_year)
    with open(os.path.join(path, fn), 'rb') as rfile:
        graph_features = pickle.load(rfile)

    graph = graph_features['graph']
    org_features = graph_features['features']
    print(graph_features.keys())

    # calculate standard deviation
    std = np.std(org_features)
    print(std)
    mean = np.mean(org_features)
    print(np.mean(org_features))

    print(graph.number_of_nodes())
    print(graph.number_of_edges())

    # print(type(org_features))
    # for case_id in range(6, 10):
    #     features = np.copy(org_features)
    #     # generate subgraph via random walk
    #     subsize = 20000
    #     current_vertex = np.random.randint(0, graph.number_of_nodes())
    #     subgraph = set()
    #     while len(subgraph) < subsize:
    #         neighbors = list(graph.neighbors(current_vertex))
    #         next_id = random.choice(neighbors)
    #         # print(len(ground_truth_subgraph))
    #         if next_id not in subgraph:
    #             subgraph.add(next_id)
    #             # print(next_id)
    #         current_vertex = next_id
    #
    #     num_blocks = 100
    #     (edge_cut, partitions) = nxmetis.partition(graph, num_blocks)
    #     node_block_dict = {}
    #     nodes_set = []
    #     for index, par in enumerate(partitions):
    #         S = graph.subgraph(par)
    #         # print(index, S.number_of_nodes(), S.number_of_edges(), nx.number_connected_components(S)) # note, !!!
    #         nodes_set.append(sorted(par))
    #         for node in par:
    #             node_block_dict[node] = index
    #
    #     block_boundary_edges_dict = {}  # key is block_id, value is a list of boundary edges (u, v), u in current block_id, v in other block
    #     cut = 0
    #     for (u, v) in graph.edges():
    #         u_block_id = node_block_dict[u]
    #         v_block_id = node_block_dict[v]
    #         if u_block_id != v_block_id:
    #             cut += 1
    #             if u_block_id not in block_boundary_edges_dict:
    #                 block_boundary_edges_dict[u_block_id] = []
    #             if v_block_id not in block_boundary_edges_dict:
    #                 block_boundary_edges_dict[v_block_id] = []
    #             block_boundary_edges_dict[u_block_id].append((u, v))
    #             block_boundary_edges_dict[v_block_id].append((v, u))
    #
    #     percent = 95
    #     x = np.percentile(org_features, [percent])[-1]
    #     print(case_id, x)
    #
    #     num_std = 2
    #     print(np.min(org_features[list(subgraph)]), np.max(org_features[list(subgraph)]))
    #     for node in graph.nodes():
    #         if node in subgraph:
    #             # print(node, features[node], end=' ')
    #             features[node] = features[node] + x
    #             # print(features[node])
    #             # features[node] = mean + num_std * std
    #
    #     print(np.min(features[list(subgraph)]), np.max(features[list(subgraph)]))
    #
    #     normzalied = False
    #     # if normzalied:
    #     #     org_features = org_features / np.max(org_features)
    #
    #     dataset = {}
    #     dataset['graph'] = graph
    #     dataset['features'] = features
    #     dataset["nodes_set"] = nodes_set
    #     dataset['true_subgraph'] = list(subgraph)
    #     dataset["block_boundary_edges_dict"] = block_boundary_edges_dict  # key is block_id, value is a list of boundary edges
    #     dataset["node_block_dict"] = node_block_dict
    #
    #
    #     # path = '/network/rit/lab/ceashpc/share_data/DBLP/1std/pkl'
    #     path = '/network/rit/lab/ceashpc/share_data/DBLP/m2std/pkl'
    #     fn = 'dataset_{}-{}_{}_{}_{}_{}_{}p_{}_new.pkl'.format(start_year, end_year, num_std, num_blocks, subsize, normzalied, percent, case_id)
    #     with open(os.path.join(path, fn), 'wb') as wfile:
    #         pickle.dump(dataset, wfile)


def generate_apdm():
    # path = '/network/rit/lab/ceashpc/share_data/DBLP/dblp_coauthor'
    start_year = 1995
    end_year = 2005
    subsize = 20000
    num_blocks = 100
    num_std = 2
    normzalied = False
    percent = 95

    for case_id in range(6, 10):
        # fn = 'dataset_{}-{}_2.pkl'.format(start_year, end_year)
        path = '/network/rit/lab/ceashpc/share_data/DBLP/m2std/pkl'
        if percent > 0:
            fn = 'dataset_{}-{}_{}_{}_{}_{}_{}p_{}_new.pkl'.format(start_year, end_year, num_std, num_blocks, subsize, normzalied, percent, case_id)
        else:
            fn = 'dataset_{}-{}_{}_{}_{}_{}.pkl'.format(start_year, end_year, num_std, num_blocks, subsize, normzalied)

        with open(os.path.join(path, fn), 'rb') as rfile:
            dataset = pickle.load(rfile)

        graph = dataset["graph"]
        features = dataset["features"]

        # todo
        print('!!!!!!!!!!!!', case_id, np.min(dataset['features'][dataset['true_subgraph']]),
              np.max(dataset['features'][dataset['true_subgraph']]))
        # dataset['features'][dataset['features'] > 10] = 10.
        print('!!!!!!!!!!!!', case_id, np.min(dataset['features'][dataset['true_subgraph']]),
              np.max(dataset['features'][dataset['true_subgraph']]))

        true_subgraph = dataset["true_subgraph"]
        sorted_features = sorted(features, reverse=True)
        for node_id in range(len(features)):
            fea = features[node_id]
            ind = sorted_features.index(fea) + 1.0
            p_value = ind / float(len(features))
            graph.node[node_id]["base"] = p_value
            graph.node[node_id]["count"] = fea


        path = '/network/rit/lab/ceashpc/share_data/DBLP/m2std/apdm'
        if percent > 0:
            fn = 'dataset_{}-{}_{}_{}_{}_{}_{}p_{}_new.apdm'.format(start_year, end_year, num_std, num_blocks, subsize, normzalied, percent, case_id)
        else:
            fn = 'dataset_{}-{}_{}_{}_{}_{}.apdm'.format(start_year, end_year, num_std, num_blocks, subsize, normzalied)
        S = graph.subgraph(true_subgraph)
        writer = apdm.APDM_Writer()
        writer.write(os.path.join(path, fn), graph, S, name=None)


def tmp():
    path = '/network/rit/lab/ceashpc/share_data/DBLP/dblp_coauthor'
    fn = 'out.dblp_coauthor'
    timestamps = []
    with open(os.path.join(path, fn), 'rb') as rfile:
        count = 0
        for line in rfile:
            if not line[0].isdigit():
                continue
            line = line.strip()
            terms = re.split('\s+', line)
            if len(terms) < 4:
                continue

            print(count)
            count += 1
            # author1 = int(terms[0])
            # author2 = int(terms[1])
            # weight = int(terms[2])
            timestamp = int(terms[3])
            timestamps.append(timestamp)

    timestamps = set(timestamps)
    print(len(timestamps))


def convert():
    start_year = 1995
    end_year = 2005
    subsize = 20000
    num_blocks = 100
    num_std = 2
    normzalied = False
    percent = 95

    dataset = []
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/app2/dblp'
    for case_id in range(10):
        fn = 'dataset_{}-{}_{}_{}_{}_{}_{}p_{}_new.pkl'.format(start_year, end_year, num_std, num_blocks, subsize, normzalied, percent, case_id)
        with open(os.path.join(path, fn), 'rb') as rfile:
            instance = pickle.load(rfile)

        dataset.append(instance)

        # print(instance.keys())

    fn = 'train.pkl'
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(dataset, wfile)


if __name__ == '__main__':

    # extract()

    # construct()

    # generate_graph_feature()

    # tmp()

    # generate_dataset()

    # generate_apdm()

    convert()