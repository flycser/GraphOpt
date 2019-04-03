#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : utils
# @Date     : 03/30/2019 18:21:41
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


from __future__ import print_function


import numpy as np


def evaluate(true_subgraphs, pred_subgraphs):
    num_time_stamps = len(true_subgraphs)

    prec_list = []
    rec_list = []
    fm_list = []
    iou_list = []

    global_intersection = 0.
    global_union = 0.

    # used for calculating performance in event interval
    true_intersection = 0.
    true_union = 0.
    true_fm_list = []


    for t in range(num_time_stamps):
        true_subgraph, pred_subgraph = set(true_subgraphs[t]), (pred_subgraphs[t])

        prec = rec = fm = iou = 0.
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
        iou_list.append(iou)

        global_intersection += len(intersection)
        global_union += len(union)
        if not 0. == len(true_subgraph):
            true_fm_list.append(fm)
            true_intersection += len(intersection)
            true_union += len(union)

    global_iou = global_intersection / float(global_union)
    true_avg_fm = np.mean(true_fm_list)
    true_global_iou = true_intersection / float(true_union)

    return np.mean(prec_list), np.mean(rec_list), np.mean(fm_list), np.mean(iou_list), global_iou, true_avg_fm, true_global_iou
    

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
