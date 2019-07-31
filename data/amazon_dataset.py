#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : amazon_dataset
# @Date     : 05/31/2019 16:03:25
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


from __future__ import print_function

import os
import re
import pickle

import numpy as np
import networkx as nx


def original():
    path = '/network/rit/lab/ceashpc/share_data/Amazon'
    fn = 'amazon-meta.txt'

    graph = nx.Graph()
    count = 0
    adj_list = {}
    feature_dict = {}
    id_dict = {}
    with open(os.path.join(path, fn)) as rfile:
        while True:
            count += 1
            line = rfile.readline()
            if not line:
                break
            if line.startswith('Id'):
                id = int(line.strip().split(':')[1].strip())

                asin = group = total = downloaded = rating = None
                similar = 0
                similar_list = []
                while True:
                    line = rfile.readline().strip()
                    # print(line.strip())
                    if len(line.strip()) == 0:
                        break

                    if line.startswith('ASIN'):
                        asin = line.split(':')[1].strip()
                        graph.add_node(asin)
                        id_dict[asin] = id
                        continue
                    if line.startswith('titile'):
                        title = line[line.find(':') + 1:].strip()
                        continue
                    if line.startswith('group'):
                        group = line.split(':')[1].strip()
                        continue
                    if line.startswith('similar'):
                        terms = re.split(r'\s+', line)
                        similar = int(terms[1])
                        if not similar == 0:
                            similar_list = terms[2:]
                            count += 1
                        continue
                    if line.startswith('categories'):
                        categories = line.split(':')[1].strip()
                        continue
                    if line.startswith('reviews'):
                        total = int(line[line.find('total:') + 7:line.find('downloaded')].strip())
                        downloaded = line[line.find('downloaded:') + 11:line.find('avg')].strip()
                        rating = line[line.find('rating:') + 7:].strip()
                        feature_dict[asin] = total
                        continue

                if asin and group:
                    adj_list[asin] = similar_list
                    print(id, asin, group, total, downloaded, rating, similar, similar_list)
                else:
                    print(id)

    for node in adj_list:
        edges = [(adj_node, node) for adj_node in adj_list[node] if adj_node in graph.nodes()]
        graph.add_edges_from(edges)

    print(graph.number_of_nodes())
    print(graph.number_of_edges())

    connected_components = [component for component in nx.connected_component_subgraphs(graph)]

    Gc = max(connected_components, key=len)
    print(len(Gc))
    subgraph = nx.subgraph(graph, Gc)
    print(subgraph.number_of_nodes())
    print(subgraph.number_of_edges())

    path = '/network/rit/lab/ceashpc/share_data/Amazon'
    fn = 'feature_map.pkl'
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(feature_dict, wfile)

    fn = 'org_graph.pkl'
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(graph, wfile)

    fn = 'connected_components.pkl'
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(connected_components, wfile)

    fn = 'id_map.pkl'
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(id_dict, wfile)

def convert_id():

    path = '/network/rit/lab/ceashpc/share_data/Amazon'
    fn = 'id_map.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        id_map = pickle.load(rfile)

    fn = 'org_graph.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        org_graph = pickle.load(rfile)

    new_graph = nx.relabel_nodes(org_graph, id_map)

    fn = 'connected_components.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        connected_components = pickle.load(rfile)

    fn = 'feature_map.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        feature_map = pickle.load(rfile)

    print(len(connected_components))
    largest_component = max(connected_components, key=len)
    connected_graph = nx.subgraph(org_graph, largest_component)
    id_graph = nx.relabel_nodes(connected_graph, id_map)
    new_id_map = {}
    id = 0
    for node in id_graph.nodes():
        new_id_map[node] = id
        id += 1

    graph = nx.relabel_nodes(id_graph, new_id_map)
    features = np.zeros(graph.number_of_nodes())
    for node in connected_graph.nodes():
        id = id_map[node]
        new_id = new_id_map[id]
        feature = feature_map[node]

        features[new_id] = feature
        print(node, id_map[node], new_id_map[id_map[node]], feature)



if __name__ == '__main__':
    # original()

    # convert_id()

    print(np.__config__.show())