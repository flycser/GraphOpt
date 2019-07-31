#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : utils
# @Date     : 03/30/2019 18:21:41
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

        logger.debug('prediction   {:d}, {}'.format(t, sorted(pred_subgraph)))
        logger.debug('ground truth {:d}, {}'.format(t, sorted(true_subgraph)))

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

def normalize_gradient(x, gradient):
    """
    make gradient [0,1] after update
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


def relabel_nodes(block_node_sets):
    """
    global_node_id - block_node_id
    :param block_node_sets:
    :return:
    """

    node_id_dict = {}
    for node_set in block_node_sets:
        ind = 0
        for node in sorted(node_set):
            node_id_dict[node] = ind
            ind +=1

    return node_id_dict

def relabel_edges(graph, block_node_sets, node_id_dict):
    """
    relabel edges of each block with block node id
    :param graph:
    :param block_node_sets:
    :param node_id_dict:
    :return:
    """
    relabeled_edge_sets = []
    for node_set in block_node_sets:
        edges = []
        for edge in nx.subgraph(graph, node_set).edges():
            node_1, node_2 = edge
            edges.append((node_id_dict[node_1], node_id_dict[node_2]))

        relabeled_edge_sets.append(edges)

    return np.array(relabeled_edge_sets)

def get_boundary_edge_x_dict(x, boundary_edges, node_id_dict):
    boundary_edge_x_dict = {}
    for edge in boundary_edges:
        node_1, node_2 = edge
        block_node_id = node_id_dict[node_1]
        adj_x_val = x[node_2]
        boundary_edge_x_dict[(block_node_id, node_2)] = adj_x_val

    return boundary_edge_x_dict