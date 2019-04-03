#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : utils
# @Date     : 03/27/2019 12:36:24
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


import numpy as np


def evaluate(true_subgraph, pred_subgraph):
    """
    calculate performance of prediction for one subgraph
    :param true_subgraph:
    :param pred_subgraph:
    :return:
    """

    true_subgraph, pred_subgraph = set(true_subgraph), set(pred_subgraph)

    prec = rec = fm = iou = 0.
    intersection = true_subgraph & pred_subgraph
    union = true_subgraph | pred_subgraph
    if not len(pred_subgraph) == 0:
        prec = len(intersection) / float(len(pred_subgraph))

    if not len(true_subgraph) == 0:
        rec = len(intersection) / float(len(true_subgraph))

    if prec + rec > 0.:
        fm = (2. * prec * rec) / (prec + rec)

    if not len(union) == 0:
        iou = len(intersection) / float(len(union))

    return prec, rec, fm, iou


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


