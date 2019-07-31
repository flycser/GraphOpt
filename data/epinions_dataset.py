#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : epinions
# @Date     : 07/01/2019 16:32:56
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :

from __future__ import print_function

import os
import re
import time
import pickle
import itertools
import operator as op
from functools import reduce
import multiprocessing


import numpy as np
import networkx as nx

from scipy.sparse import *

def convert_network():
    graph = nx.DiGraph()

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions'
    fn = 'trust_data.txt'
    count = 0
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            terms = re.split('\s+', line.strip())
            node_1 = int(terms[0])
            node_2 = int(terms[1])
            # print(count, node_1, node_2)
            count += 1
            graph.add_edge(node_1, node_2)

    print(graph.number_of_nodes())
    print(graph.number_of_edges())
    largest_cc = max(nx.weakly_connected_components(graph), key=len)
    largest_subgraph = nx.subgraph(graph, largest_cc)
    print(largest_subgraph.number_of_nodes())
    print(largest_subgraph.number_of_edges())

    sorted_nodes = sorted(graph.nodes())
    fn = 'map_user.pkl'
    map_user = {}
    max_id = 0
    for id, node in enumerate(sorted_nodes):
        map_user[node] = id
        if id > max_id:
            max_id = id
    print(max_id)

    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(map_user, wfile)

    graph = nx.relabel_nodes(graph, map_user)
    fn = 'graph.pkl'
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(graph, wfile)


def similarity():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions'
    # fn = 'graph.pkl'
    # with open(os.path.join(path, fn), 'rb') as rfile:
    #     graph = pickle.load(rfile)

    fn = 'map_user.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        map_user = pickle.load(rfile)

    fn = 'ratings_data.txt'
    items = []
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            terms = re.split('\s+', line.strip())
            user = int(terms[0])
            if user not in map_user:
                print(user)
                continue
            item = int(terms[1])
            items.append(item)
            # rate = int(terms[2])

    fn = 'map_item.pkl'
    sorted_items = sorted(list(set(items)))
    map_item = {}
    for id, item in enumerate(sorted_items):
        map_item[item] = id

    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(map_item, wfile)


def construct_rate_matrix():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions'
    fn = 'graph.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        graph = pickle.load(rfile)

    fn = 'map_user.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        map_user = pickle.load(rfile)

    fn = 'map_item.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        map_item = pickle.load(rfile)

    print(len(map_user), len(map_item))
    # rating_array = np.zeros(shape=(len(map_user), len(map_item)), dtype=np.int) # memory error
    rating_array = lil_matrix((len(map_user), len(map_item)), dtype='i')
    print(rating_array.shape)
    fn = 'ratings_data.txt'
    ratings = set()
    users = set()
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            terms = re.split('\s+', line.strip())
            user = int(terms[0])
            users.add(user)
            if user not in map_user:
                # print(user)
                continue
            item = int(terms[1])
            rate = int(terms[2])
            if (map_user[user], map_item[item]) in ratings:
                print('xxxxxxxxxxx', user, item)
            else:
                rating_array[map_user[user], map_item[item]] = rate
                ratings.add((map_user[user], map_item[item]))

    print(len(users))

    fn = 'ratings_array.pkl'
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(rating_array, wfile)

    print(rating_array.shape)


def calculate_correlation():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions'
    fn = 'ratings_array.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        rating_array = pickle.load(rfile)

    print('xxxxxvvvvvvvvv')

    print(rating_array.shape)

    fn = 'map_user.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        map_user = pickle.load(rfile)

    fn = 'ratings_data.txt'
    users = set()
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            terms = re.split('\s+', line.strip())
            user = int(terms[0])
            if user not in map_user:
                # print(user)
                continue
            users.add(map_user[user])

    print(len(users))

    fn = 'graph.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        graph = pickle.load(rfile)

    print(max(list(graph.nodes())))

    mean_rate = {}
    correlations = {}
    count = 0

    filtered_users = []
    for user in users:
        rate = rating_array[user, :]
        if rate.getnnz() > 5:
            filtered_users.append(user)

    print(len(filtered_users))

    for user_i, user_f in itertools.combinations(filtered_users, 2):

        rate_i = rating_array[user_i, :] # sparse format
        rate_f = rating_array[user_f, :]

        nnz_i = rate_i.nonzero()[1]
        nnz_f = rate_f.nonzero()[1]

        if count % 10000 == 0:
            print(count, time.time())
        count += 1

        common_entries = np.intersect1d(nnz_i, nnz_f)
        d = len(common_entries)
        if d <= 5:
            continue

        # if rate_i.getnnz() <=5 or rate_f.getnnz() <=5:
        #     continue

        x_i = rate_i.toarray().reshape((-1, )) # dense format
        x_f = rate_f.toarray().reshape((-1, ))

        # nnz_i = x_i.nonzero()[0]
        # nnz_f = x_f.nonzero()[0]


        # if len(nnz_i) == 0 or len(nnz_f) == 0:
        #     continue
        # print(count, len(nnz_i), len(nnz_f))
        if user_i not in mean_rate:
            # mean_rate[user_i] = np.true_divide(np.sum(x_i), len(nnz_i))
            mean_rate[user_i] = np.true_divide(rate_i.sum(), len(nnz_i))
        if user_f not in mean_rate:
            # mean_rate[user_f] = np.true_divide(np.sum(x_f), len(nnz_f))
            mean_rate[user_f] = np.true_divide(rate_f.sum(), len(nnz_f))

        if d > 5:
            # print(user_i, user_f)
            # print(nnz_i)
            # print(x_i[nnz_i])
            # print(nnz_f)
            # print(x_f[nnz_f])
            # print(common_entries)
            # print(x_i[common_entries])
            # print(x_f[common_entries])
            a = x_i[common_entries] - mean_rate[user_i]
            b = x_f[common_entries] - mean_rate[user_f]

            # print(mean_rate[user_i], mean_rate[user_f])
            x = np.inner(a, b)
            y = np.inner(a, a)
            z = np.inner(b, b)
            w = x / (np.sqrt(y) * np.sqrt(z))
            correlations[(user_i, user_f)] = w
            # print(x, y, w)
            # print(a)
            # print(x_i[nnz_i])
            # print(b)
            # print(x_f[nnz_f])

    print(len(correlations))
    fn = 'correlations_2.pkl'
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(correlations, wfile)


def similarity_mps():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions'
    fn = 'ratings_array.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        rating_array = pickle.load(rfile)

    print(rating_array.shape)

    fn = 'map_user.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        map_user = pickle.load(rfile)

    fn = 'ratings_data.txt'
    users = set()
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            terms = re.split('\s+', line.strip())
            user = int(terms[0])
            if user not in map_user:
                # print(user)
                continue
            users.add(map_user[user])

    print(len(users))

    fn = 'graph.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        graph = pickle.load(rfile)

    print(max(list(graph.nodes())))

    filtered_users = []
    for user in users:
        rate = rating_array[user, :]
        if rate.getnnz() >= 5:
            filtered_users.append(user)

    print(len(filtered_users))

    n = len(filtered_users)
    r = 2
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    max_pairs = numer / denom

    input_paras = []
    sep = 10000000
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions/leq5'
    for start in np.arange(0, max_pairs, sep):
        end = start + sep if start + sep < max_pairs else max_pairs
        paras = path, rating_array, filtered_users, start, end
        input_paras.append(paras)

    num_prcs = 30
    pool = multiprocessing.Pool(processes=num_prcs)
    pool.map(calculate, input_paras)
    pool.close()
    pool.join()


def calculate(paras):
    path, rating_array, filtered_users, start, end = paras

    count = -1
    mean_rate = {}
    correlations = {}
    print(start, end)
    for user_i, user_f in itertools.combinations(filtered_users, 2):

        count += 1
        if count < start:
            continue
        elif count >= end:
            break

        rate_i = rating_array[user_i, :] # sparse format
        rate_f = rating_array[user_f, :]

        nnz_i = rate_i.nonzero()[1]
        nnz_f = rate_f.nonzero()[1]

        if count % 100000 == 0:
            print(count, time.time())
        # count += 1

        common_entries = np.intersect1d(nnz_i, nnz_f)
        d = len(common_entries)
        if d < 5:
            continue

        x_i = rate_i.toarray().reshape((-1, )) # dense format
        x_f = rate_f.toarray().reshape((-1, ))

        if user_i not in mean_rate:
            # mean_rate[user_i] = np.true_divide(np.sum(x_i), len(nnz_i))
            mean_rate[user_i] = np.true_divide(rate_i.sum(), len(nnz_i))
        if user_f not in mean_rate:
            # mean_rate[user_f] = np.true_divide(np.sum(x_f), len(nnz_f))
            mean_rate[user_f] = np.true_divide(rate_f.sum(), len(nnz_f))

        # print(user_i, user_f)
        # print(nnz_i)
        # print(x_i[nnz_i])
        # print(nnz_f)
        # print(x_f[nnz_f])
        # print(common_entries)
        # print(x_i[common_entries])
        # print(x_f[common_entries])
        a = x_i[common_entries] - mean_rate[user_i]
        b = x_f[common_entries] - mean_rate[user_f]

        # print(mean_rate[user_i], mean_rate[user_f])
        x = np.inner(a, b)
        y = np.inner(a, a)
        z = np.inner(b, b)
        w = x / (np.sqrt(y) * np.sqrt(z))
        correlations[(user_i, user_f)] = w
        # print(x, y, w)
        # print(a)
        # print(x_i[nnz_i])
        # print(b)
        # print(x_f[nnz_f])

    print(len(correlations))
    fn = 'correlations_{}_{}.pkl'.format(start, end)
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(correlations, wfile)

def get_correlations():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions/leq5'
    sep = 10000000
    count = 0
    correlations = {}
    for start in range(0, 270316126, sep):
        end = start + sep if start + sep < 270316126 else 270316126
        fn = 'correlations_{}_{}.pkl'.format(start, end)
        with open(os.path.join(path, fn), 'rb') as rfile:
            part_correlations = pickle.load(rfile)
            correlations.update(part_correlations)
            print(len(part_correlations))
            count += len(part_correlations)

    print(count)
    print(len(correlations))

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions'
    fn = 'correlations_mps.pkl'
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(correlations, wfile)


def generate_DCS_files():
    rpath = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions'
    wpath = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions/dcs_test'

    fn = 'graph.pkl'
    with open(os.path.join(rpath, fn), 'rb') as rfile:
        graph = pickle.load(rfile)

    fn = '01Nodes.txt'
    idx = 0
    map_node = {}
    with open(os.path.join(wpath, fn), 'w') as wfile:
        for node in graph.nodes():
            wfile.write('{}\n'.format(node))
            map_node[node] = idx
            idx += 1

    fn = '02EdgesP.txt'
    with open(os.path.join(wpath, fn), 'w') as wfile:
        for edge in graph.edges():
            node_1, node_2 = tuple(sorted(list(edge)))
            wfile.write('{} {}\n'.format(map_node[node_1], map_node[node_2]))

    fn = 'correlations_mps.pkl'
    with open(os.path.join(rpath, fn), 'rb') as rfile:
        correlations = pickle.load(rfile)

    fn = '03EdgesC.txt'
    fn_2 = '03EdgesC_non_normalize.txt'
    with open(os.path.join(wpath, fn), 'w') as wfile, open(os.path.join(wpath, fn_2), 'w') as wfile_2:
        for edge in correlations:
            node_1, node_2 = tuple(sorted(list(edge))) # sorted edge
            weight = correlations[edge]
            wfile.write('{} {} {:.5f}\n'.format(map_node[node_1], map_node[node_2], (weight+1)/2))
            wfile_2.write('{} {} {:.5f}\n'.format(map_node[node_1], map_node[node_2], weight))


def generate_attributes():
    rpath = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions'
    wpath = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions'

    fn = 'graph.pkl'
    with open(os.path.join(rpath, fn), 'rb') as rfile:
        graph = pickle.load(rfile)

    num_nodes = graph.number_of_nodes()
    # random walk
    start_node = next_node = np.random.choice(range(num_nodes))
    subgraph = set()
    subgraph.add(start_node)
    restart = 0.1
    count = 1000
    while True:
        if len(subgraph) >= count:
            break

        successors = [node for node in nx.neighbors(graph, next_node)]
        predecessor = [node for node in nx.predecessor(graph, next_node)]
        neighbors = successors + predecessor # note, python extend not return self

        if np.random.uniform() > restart:
            next_node = np.random.choice(neighbors)
        else: # restart
            next_node = start_node

        subgraph.add(next_node)
        print(len(subgraph))

    mean_1 = 5.
    mean_2 = 0.
    std = 1.
    attributes = np.zeros(graph.number_of_nodes())
    for node in graph.nodes():
        if node in subgraph:
            attributes[node] = np.random.normal(mean_1, std)
        else:
            attributes[node] = np.random.normal(mean_2, std)

    fn = 'attributes.pkl'
    with open(os.path.join(wpath, fn), 'wb') as wfile:
        pickle.dump({'attributes':attributes, 'subgraph':subgraph}, wfile)

def calculate_ems():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions'
    fn = 'graph.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        graph = pickle.load(rfile)

    fn = 'attributes.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        attributes = pickle.load(rfile)['attributes']

    for idx in range(10):
        path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions/feng/Output{}'.format(idx)
        fn = '01DenseSubgraphNodes.txt'
        subgraph = set()
        with open(os.path.join(path, fn)) as rfile:
            for line in rfile:
                node = int(line.strip())
                subgraph.add(node)

        subgraph = list(subgraph)

        x = np.zeros(graph.number_of_nodes())
        x[subgraph] = 1.

        print(len(np.nonzero(x)[0]))
        ems = np.dot(attributes, x) / np.sqrt(np.sum(x))
        print(ems)

def generate_subgraph_biased_random_walk(id=0):
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions'
    fn = 'graph.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        graph = pickle.load(rfile)

    print(graph.number_of_nodes())
    print(graph.number_of_edges())

    print(len([1 for (u, v) in graph.edges() if u in graph[v]]))

    undigraph = graph.to_undirected() # directed graph to undirected graph
    print(undigraph.number_of_nodes())
    print(undigraph.number_of_edges())
    print(nx.is_connected(undigraph))

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions/dcs_test'
    fn = '03EdgesC.txt'
    correlations = {} # edge weight of conceptual network
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            terms = line.strip().split(' ')
            node_1, node_2, weight = int(terms[0]), int(terms[1]), float(terms[2])

            correlations[(node_1, node_2)] = weight

    num_nodes = undigraph.number_of_nodes()
    node_degree = np.zeros(num_nodes)
    for current_node in undigraph.nodes():
        sum_weight = 0.
        for node in nx.neighbors(undigraph, current_node):
            pair = tuple(sorted(list((node, current_node))))
            weight = correlations[pair] if pair in correlations else 0.
            sum_weight += weight

        node_degree[current_node] = sum_weight

    # random walk
    start_node = next_node = np.random.choice(range(num_nodes))
    subgraph = set()
    subgraph.add(start_node)
    restart = 0.1
    count = 1000
    print(start_node)
    # random walk on undirected physical graph, biased for nodes with higher degree on conceptual network
    while True:
        if len(subgraph) >= count:
            break

        neighbors = [node for node in nx.neighbors(undigraph, next_node)]

        neighbor_degree_dist = [node_degree[node] for node in neighbors]
        sum_prob = np.sum(neighbor_degree_dist)
        # note, when there are no neighbors for one node on conceptual network, its probabilities to other neighbor nodes are equal
        normalized_prob_dist = [prob / sum_prob if not sum_prob == 0. else 1. / len(neighbors) for prob in neighbor_degree_dist]
        # if sum_prob == 0.:
        #     print(len(neighbors))
        #     print(neighbor_degree_dist)
        #     print(sum_prob)
        #     print(normalized_prob_dist)
        #     break
        # print(len(neighbor_degree_dist))
        # print(normalized_prob_dist)

        if np.random.uniform() > restart:
            next_node = np.random.choice(neighbors, p=normalized_prob_dist) # biased for those nodes with high degree
        else:  # restart
            next_node = start_node

        subgraph.add(next_node)
        print(len(subgraph))

    if len(subgraph) < count:
        print('generation fails')
        return

    mean_1 = 5.
    mean_2 = 0.
    std = 1.
    attributes = np.zeros(graph.number_of_nodes())
    for node in graph.nodes():
        if node in subgraph:
            attributes[node] = np.random.normal(mean_1, std)
        else:
            attributes[node] = np.random.normal(mean_2, std)

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions/syn'
    fn = 'attributes_biased_{}.pkl'.format(id)
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump({'attributes': attributes, 'subgraph': subgraph}, wfile)


def calculate_subgraph_density():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions/dcs_test'
    fn = '03EdgesC.txt'
    correlations = {}  # edge weight of conceptual network
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            terms = line.strip().split(' ')
            node_1, node_2, weight = int(terms[0]), int(terms[1]), float(terms[2])

            correlations[(node_1, node_2)] = weight

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions'
    fn = 'graph.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        graph = pickle.load(rfile)

    undigraph = nx.to_undirected(graph)

    fn = 'attributes.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        dataset = pickle.load(rfile)

    subgraph = nx.subgraph(undigraph, dataset['subgraph'])
    density = nx.density(subgraph)
    print(density)
    sum_weight = 0.
    for u, v in subgraph.edges():
        pair = tuple(sorted(list((u, v))))
        weight = correlations[pair] if pair in correlations else 0.
        sum_weight += weight
    print(sum_weight / subgraph.number_of_nodes())


    id = 9
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions/syn'
    fn = 'attributes_biased_{}.pkl'.format(id)
    with open(os.path.join(path, fn), 'rb') as rfile:
        dataset = pickle.load(rfile)

    subgraph = nx.subgraph(undigraph, dataset['subgraph'])
    density = nx.density(subgraph)
    print(density)
    sum_weight = 0.
    for u, v in subgraph.edges():
        pair = tuple(sorted(list((u, v))))
        weight = correlations[pair] if pair in correlations else 0.
        sum_weight += weight
    print(sum_weight / subgraph.number_of_nodes())


def generate_dcs_amen_dataset():

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions'
    fn = 'graph.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        graph = pickle.load(rfile)

    id = 0
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions/syn'
    fn = 'attributes_biased_{}.pkl'.format(id)
    with open(os.path.join(path, fn), 'rb') as rfile:
        dataset = pickle.load(rfile)

    attributes = dataset['attributes']
    subgraph = dataset['subgraph']
    # print(type(subgraph))
    # print(sorted(list(subgraph)))

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions/syn/amen'
    fn = 'instance_{}.txt'.format(id)
    with open(os.path.join(path, fn), 'w') as wfile:
        wfile.write('{} {}\n'.format(graph.number_of_nodes(), 1))
        for node in graph.nodes():
            wfile.write('{}\n'.format(attributes[node]))

        wfile.write('{}\n'.format(graph.number_of_edges()))
        for node_1, node_2 in graph.edges():
            node_1, node_2 = (node_1, node_2) if node_1 < node_2 else (node_2, node_1)
            wfile.write('{} {} 1.0\n'.format(node_1, node_2))

        wfile.write(' '.join([str(node) for node in sorted(list(subgraph))]) + '\n')
        wfile.write('0')




if __name__ == '__main__':

    generate_dcs_amen_dataset()

    # for id in range(10):
    #     generate_subgraph_biased_random_walk(id)

    # calculate_subgraph_density()

    # convert_network()

    # similarity()

    # construct_rate_matrix()

    # calculate_correlation()

    # similarity_mps()

    # get_correlations()

    # generate_DCS_files()

    # generate_attributes()

    # path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions'
    # fn = 'correlations_mps.pkl'
    # with open(os.path.join(path, fn), 'rb') as rfile:
    #     correlations = pickle.load(rfile)
    #
    # print(len(correlations))

    # path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions'
    # fn = 'attributes.pkl'
    # with open(os.path.join(path, fn), 'rb') as rfile:
    #     attributes = pickle.load(rfile)
    #
    # print(attributes['subgraph'])
    #
    # seeds = np.random.choice(list(attributes['subgraph']), size=10, replace=False)
    # print(seeds)

    # path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions/feng'
    # fn = 'seeds.txt'
    # with open(os.path.join(path, fn)) as rfile:
    #     for idx, line in enumerate(rfile):
    #         seed = line.strip()
    #         output = os.path.join(path, 'Output{}'.format(idx))
    #         os.mkdir(output, 0744)
    #         wfn = '04Seeds{}.txt'.format(idx)
    #         with open(os.path.join(path, wfn), 'w') as wfile:
    #             wfile.write(seed)

    # calculate_ems()

    # relabeled_edges_set = [[(1, 2), (2, 3)], [(0, 1), (3, 4), (1, 3)]]
    #
    # x = np.array(relabeled_edges_set)
    #
    # print(x)
    # print(type(x))