#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : greedyAP_ba_exp
# @Date     : 08/07/2019 12:56:02
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


from __future__ import print_function

import os
import sys
import re
import time
import pickle
import collections

from random import shuffle

import numpy as np
import networkx as nx


def greedAP(graph, weight, lmbd, normalize=True, sort=False, verbose=True):

    # normalize weight
    if normalize:
        weight = (weight - np.min(weight)) / (np.max(weight) - np.min(weight))
    else:
        weight = weight - np.min(weight) # note, positive

    if sort:
        # weight[::-1].sort() # error
        sorted_nodes = np.argsort(weight)[::-1] # desendingly

    nodes = [node for node in graph.nodes()]
    shuffle(nodes)

    X = set() # start from empty set

    start_time = time.time()
    # add node util marginal gain is negative
    X.add(nodes[0])
    i = 1
    while True:
        if i >= len(nodes):
            break

        u = nodes[i]
        shortest_lengths_from_u = nx.shortest_path_length(graph, source=u)
        Dap = 0
        for node in X:
            Dap -= shortest_lengths_from_u[node]

        margin_gain = lmbd * weight[u] + Dap

        if margin_gain < 0:
            if verbose:
                print('stop at node {:d}, margin gain: {:.5f}'.format(u, margin_gain))
            break
        else:
            X.add(u)
            if verbose:
                print('add node {:d}, node weight {:.2f}, margin gain: {:.5f}'.format(u, weight[u], margin_gain))

        i += 1

    if verbose:
        print('run time: {:.5f} sec.'.format(time.time() - start_time))

    return X


def BFNS(graph, weight, lmbd, shortest_path_lengths, start_node, lcc_diameter, normalize=True, sort=True, verbose=True):
    # normalize weight
    if normalize:
        weight = (weight - np.min(weight)) / (np.max(weight) - np.min(weight))
    else:
        weight = weight - np.min(weight)

    if sort:
        # weight[::-1].sort() # error
        sorted_nodes = np.argsort(weight)[::-1] # desendingly


    nodes = [node for node in graph.nodes()]
    shuffle(nodes)

    X = set() # start from empty set
    Y = set([node for node in graph.nodes()]) # start from entire node set

    if start_node > 0:
        X.add(start_node)
        Y.remove(start_node)

    start_time = time.time()

    for i in range(graph.number_of_nodes()):
        # how to select next node? randomly? shuffle node list firstly
        u = nodes[i]
        if u == start_node:
            continue

        # shortest_lengths_from_u = nx.shortest_path_length(graph, source=u)
        shortest_lengths_from_u = shortest_path_lengths[u]

        #   Q_AP(X\cup u) - Q_AP(X)
        # = I(X\cup u) + D_AP(V) - D_AP(X\cup u) - I(X) - D_AP(V) + D_AP(X)
        # = I(X) + I(u) + D_AP(V) - D_AP(X\cup u) - I(X) - D_AP(V) + D_AP(X)
        # = I(u) + D(X) - D_AP(X\cup u)
        Dap_a = 0
        for node in X:
            if node not in shortest_lengths_from_u.keys(): # note, handle not connected nodes
                # Dap_a -= max(shortest_lengths_from_u.values())
                Dap_a -= (max(shortest_lengths_from_u.values()) + lcc_diameter)
                print('tedhasjkdhasjkhdkashdjkash')
                return
            else:
                Dap_a -= shortest_lengths_from_u[node]
        a = lmbd * weight[u] + Dap_a

        #   Q_AP(Y\u) - Q_AP(Y)
        # = I(Y\u) + D_AP(V) - D_AP(Y\u) - I(Y) - D_AP(V) + D_AP(Y)
        # = I(Y) - I(u) + D_AP(V) - D_AP(Y\u) - I(Y) - D_AP(V) + D_AP(Y)
        # = -I(u) + D_AP(Y) - D_AP(Y\u)
        Dap_b = 0
        for node in Y:
            if node not in shortest_lengths_from_u.keys(): # note, handle not connected nodes
                # Dap_b += max(shortest_lengths_from_u.values())
                Dap_b += (max(shortest_lengths_from_u.values()) + lcc_diameter)
                print('tedhasjkdhasjkhdkashdjkash')
                return
            else:
                Dap_b += shortest_lengths_from_u[node]

        b = - lmbd * weight[u] + Dap_b

        a = a if a > 0 else 0
        b = b if b > 0 else 0

        if np.random.uniform() < a / (a+b):
            X.add(u)
            if verbose:
                print('add node {:d} to X, margin gain: {:.5f}'.format(u, a))
        else:
            Y.remove(u)
            if verbose:
                print('remove node {:d} from Y, margin gain: {:.5f}'.format(u, b))

    if verbose:
        print('run time: {:.5f} sec.'.format(time.time() - start_time))

    # print(sorted(list(X)))
    # print(sorted(list(Y)))

    return X

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

def run_instance(case_id, procession):

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/Mus_Multiplex_Genetic/dual'
    fn = 'dual_mus_case_{}.pkl'.format(case_id)
    with open(os.path.join(path, fn), 'rb') as rfile:
        instance = pickle.load(rfile)

    first_graph = instance['first_graph']
    second_graph = instance['second_graph']

    weight = instance['weight']
    true_subgraph = instance['true_subgraph']
    diameter = nx.diameter(first_graph)

    start_time = time.time()
    shortest_path_lengths = {}
    for node in first_graph.nodes():
        x = nx.shortest_path_length(first_graph, source=node)
        shortest_path_lengths[node] = x
    print('run time shortest path {}'.format(time.time() - start_time))

    results = []
    times = 2
    for lmbd in procession:
        for i in range(times):
            # x = greedAP(first_graph, weight, lmbd, normalize=True, sort=False)
            print('lambda {}, time {}'.format(lmbd, i))
            x = BFNS(first_graph, weight, lmbd, shortest_path_lengths, start_node=-1, lcc_diameter=diameter, normalize=True, sort=False, verbose=False)

            density = nx.density(nx.subgraph(second_graph, x))
            p, r, f, i = evaluate(true_subgraph, x)
            results.append((p, r, f, i, density))

    results = np.array(results)
    print(results)

def decide_lambda():
    path = '/network/rit/lab/ceashpc/fjie/projects/GraphOpt/block_ghtp'
    fn = 'slurm-238374.out'
    results = {}
    with open(os.path.join(path, fn), 'rb') as rfile:
        while True:
            line = rfile.readline()
            if not line:
                break

            if line.startswith('[['):
                count = 1
                while True:

                    if line.strip().startswith('[['):
                        terms = re.split('\s+', line.strip()[2:-1].strip())
                        # print(terms)
                        p, r, f, i = float(terms[0]), float(terms[1]), float(terms[2]), float(terms[3])
                        if count not in results:
                            results[count] = []
                            results[count].append((p, r, f, i))
                        else:
                            results[count].append((p, r, f, i))

                        count += 1

                    elif line.strip().endswith(']]'):
                        terms = re.split('\s+', line.strip()[1:-2].strip())
                        # print(terms)
                        p, r, f, i = float(terms[0]), float(terms[1]), float(terms[2]), float(terms[3])
                        if count not in results:
                            results[count] = []
                            results[count].append((p, r, f, i))
                        else:
                            results[count].append((p, r, f, i))
                        break
                    else:
                        terms = re.split('\s+', line.strip()[1:-1].strip())
                        # print(terms)
                        p, r, f, i = float(terms[0]), float(terms[1]), float(terms[2]), float(terms[3])
                        if count not in results:
                            results[count] = []
                            results[count].append((p, r, f, i))
                        else:
                            results[count].append((p, r, f, i))

                        count += 1

                    line = rfile.readline()
                    if not line:
                        break

    print(results)
    for k in results:
        print(k)
        tmp_result = results[k]
        tmp_result = np.array(tmp_result)
        print(tmp_result)
        print(tmp_result.mean(axis=0))


if __name__ == '__main__':
    pass

    # case_id = 0
    case_id = int(sys.argv[1])
    print('case {}'.format(case_id))
    procession = range(1, 5002, 500)
    run_instance(case_id, procession)

    # decide_lambda()