#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : utils
# @Date     : 04/05/2019 17:03:41
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

import numpy as np
import networkx as nx


def evaluate_evo(true_subgraphs, pred_subgraphs):
    num_time_stamps = len(true_subgraphs)

    prec_list = []
    rec_list = []
    fm_list = []
    iou_list = []

    global_intersection = 0.
    global_union = 0.

    # used for calculating performance in signal interval
    valid_intersection = 0.
    valid_union = 0.
    num_pred_nodes = 0.
    num_valid_pred_nodes = 0.
    num_valid_nodes = 0.
    vaild_fm_list = []

    for t in range(num_time_stamps):
        true_subgraph, pred_subgraph = set(true_subgraphs[t]), set(pred_subgraphs[t])

        intersection = true_subgraph & pred_subgraph
        union = true_subgraph | pred_subgraph

        if not 0. == len(pred_subgraph):
            prec = len(intersection) / float(len(pred_subgraph))
        else:
            prec = 1.
        prec_list.append(prec)

        if not 0. == len(true_subgraph):
            rec = len(intersection) / float(len(true_subgraph))
        else:
            rec = 1.
        rec_list.append(rec)

        if prec + rec > 0.:
            fm = (2. * prec * rec) / (prec + rec)
        else:
            fm = 0.
        fm_list.append(fm)

        if not 0. == len(union):
            iou = len(intersection) / float(len(union))
        else:
            iou = 1.
        iou_list.append(iou)

        global_intersection += len(intersection)
        global_union += len(union)

        if not 0. == len(true_subgraph):
            vaild_fm_list.append(fm)
            valid_intersection += len(intersection)
            valid_union += len(union)
            num_valid_pred_nodes += len(pred_subgraph)

        num_pred_nodes += len(pred_subgraph)
        num_valid_nodes += len(true_subgraph)

        # logger.debug('prediction   {:d}, {}'.format(t, sorted(pred_subgraph)))
        # logger.debug('ground truth {:d}, {}'.format(t, sorted(true_subgraph)))

    global_prec = global_rec = global_fm = global_iou = valid_global_iou = 0.
    if not 0. == global_union:
        global_prec = global_intersection / float(num_pred_nodes)
        global_rec = global_intersection / float(num_valid_nodes)
        global_fm = 0.
        if global_prec + global_rec > 0.:
            global_fm = 2 * global_prec * global_rec / (global_prec + global_rec)
        global_iou = global_intersection / float(global_union)
    valid_avg_fm = np.mean(vaild_fm_list)
    valid_global_prec = valid_intersection / float(num_valid_pred_nodes)
    valid_global_rec = valid_intersection / float(num_valid_nodes)
    valid_global_fm = 0.
    if valid_global_prec + valid_global_rec > 0.:
        valid_global_fm = 2 * valid_global_prec * valid_global_rec / (valid_global_prec + valid_global_rec)
    if not 0. == valid_union:
        valid_global_iou = valid_intersection / float(valid_union)

    return global_prec, global_rec, global_fm, global_iou, valid_global_prec, valid_global_rec, valid_global_fm, valid_global_iou, valid_avg_fm, np.mean(prec_list), np.mean(rec_list), np.mean(fm_list), np.mean(iou_list)


def evaluate_block(true_subgraph, pred_subgraph):
    true_subgraph, pred_subgraph = set(list(true_subgraph)), set(list(pred_subgraph)) # note, list could convert ndarray to list
    intersection = true_subgraph.intersection(pred_subgraph)
    union = true_subgraph.union(pred_subgraph)

    if len(true_subgraph) == 0:
        return 0, 0, 0, 0

    if not len(pred_subgraph) == 0:
        prec = len(intersection) / float(len(pred_subgraph))
    else:
        prec = 0.

    rec = len(intersection) / float(len(true_subgraph))

    if prec + rec > 0.:
        fm = (2. * prec * rec) / (prec + rec)
    else:
        fm = 0.

    iou = len(intersection) / float(len(union))

    return prec, rec, fm, iou


def normalize_gradient(x, gradient):
    """
    make gradient [0,1] after update for maximization
    :param x:
    :param gradient:
    :return:
    """

    normalized_gradient = np.zeros_like(gradient)
    for i in range(len(gradient)):
        if 0. < gradient[i] and 1. == x[i]:
            normalized_gradient[i] = 0.
        elif 0. > gradient[i] and 0. == x[i]:
            normalized_gradient[i] = 0.
        else:
            normalized_gradient[i] = gradient[i]

    return normalized_gradient

def normalize(x):
    normalized_x = np.zeros_like(x)
    for i in range(len(x)):
        if 0. > x[i]:
            normalized_x[i] = 0.
        elif 1. < x[i]:
            normalized_x[i] = 1.
        else:
            normalized_x[i] = x[i]

    return normalized_x

def post_process_evo(graph, raw_pred_subgraphs, dataset=None):
    node_global2local_mapping = {}  # regenerated node id - time stamp, original node id
    node_local2global_mapping = {}  # time stamp, original node id - regenerated node id
    num_nodes = graph.number_of_nodes()
    num_time_stamps = len(raw_pred_subgraphs)
    edges = []

    nodes = []
    node_id = 0
    # renumbered all nodes in raw predicted subgraphs
    for t in range(num_time_stamps):
        current_pred_subgraph = raw_pred_subgraphs[t]
        # print(t, nx.is_connected(nx.subgraph(graph, current_pred_subgraph)))
        # print(max(nx.connected_components(nx.subgraph(graph, current_pred_subgraph)), key=len))
        if not set(current_pred_subgraph):
            continue
        else:
            for node in current_pred_subgraph:
                nodes.append(node_id)
                node_global2local_mapping[node_id] = (t, node)
                node_local2global_mapping[(t, node)] = node_id
                node_id += 1

    # add edges with renumbered node id
    for t in range(num_time_stamps):
        current_pred_subgraph = nx.subgraph(graph, raw_pred_subgraphs[t])
        # add each time stamps / local edges
        edges.extend(
            [(node_local2global_mapping[(t, node_1)], node_local2global_mapping[(t, node_2)]) for node_1, node_2 in
             current_pred_subgraph.edges()])
        # add edge between two consecutive time stamps
        if t > 0:
            prev_pred_subgraph_nodes = set(raw_pred_subgraphs[t - 1])
            cur_pred_subgraph_nodes = set(raw_pred_subgraphs[t])
            overlap_nodes = prev_pred_subgraph_nodes & cur_pred_subgraph_nodes
            for node in overlap_nodes:
                edges.append((node_local2global_mapping[(t - 1, node)], node_local2global_mapping[(t, node)]))

    combined_graph = nx.Graph()
    combined_graph.add_edges_from(edges)
    if 'dc' == dataset or 'bj' == dataset:
        largest_ccs_candidates = [sorted(cc) for cc in nx.connected_components(combined_graph) if len(cc) > 5] # note, only keep those components whose number of nodes > 5

        refined_pre_subgraphs = [[] for _ in range(num_time_stamps)]
        for cc in largest_ccs_candidates:
            for node in cc:
                t, local_node_id = node_global2local_mapping[node]
                refined_pre_subgraphs[t].append(local_node_id)
    else:
        largest_cc = max(nx.connected_components(combined_graph), key=len)

        refined_pre_subgraphs = [[] for _ in range(num_time_stamps)]
        for node in largest_cc:
            t, local_node_id = node_global2local_mapping[node]
            refined_pre_subgraphs[t].append(local_node_id)

    return refined_pre_subgraphs

def post_process_block(graph, raw_pred_subgraph, dataset=None):

    subgraph = nx.subgraph(graph, raw_pred_subgraph)
    largest_cc = max(nx.connected_components(subgraph), key=len)
    refined_pred_subgraph = sorted([node for node in largest_cc])

    return refined_pred_subgraph

def relabel_nodes(nodes_set):
    """
    relabel nodes in a block with local ids
    :param nodes_set:
    :return:
    """

    nodes_id_dict = {}
    for block_nodes in nodes_set:
        ind = 0
        for node in sorted(block_nodes): # relabel nodes in one block
            nodes_id_dict[node] = ind
            ind += 1

    return nodes_id_dict

def relabel_edges(graph, nodes_set, nodes_id_dict):
    # relabel edges with local ids
    relabeled_edges_set = []
    for block_nodes in nodes_set:
        edges = []
        for u, v in nx.subgraph(graph, block_nodes).edges():
            edges.append((nodes_id_dict[u], nodes_id_dict[v]))

        relabeled_edges_set.append(edges)

    return np.array(relabeled_edges_set)

def get_boundary_xs(x, block_boundary_edges, nodes_id_dict):
    """
    get x value on the other end of edge across two blocks
    :param x:
    :param block_boundary_edges:
    :param nodes_id_dict:
    :return:
    """

    boundary_xs_dict = {}
    for (u, v) in block_boundary_edges:
        nodes_id_in_block = nodes_id_dict[u]
        adj_x_val = x[v]
        boundary_xs_dict[(nodes_id_in_block, v)] = adj_x_val

    return  boundary_xs_dict