#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : objs
# @Date     : 03/27/2019 23:01:33
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


from __future__ import print_function

import sys

import numpy as np
import networkx as nx


class EMS(object):

    def __init__(self, features, graph):

        self.features = features
        self.num_nodes = len(features)
        self.graph = graph

    def get_obj_val(self, x):

        func_val = 0.
        sum_x = np.sum(x)
        if 0. < sum_x:
            ct_x = np.dot(self.features, x)
            func_val = ct_x / np.sqrt(sum_x)

        return func_val

    def get_init_x_random(self):
        init_x = np.random.rand(self.num_nodes)
        init_x = np.array(init_x >= 0.5, dtype=np.float64)

        return init_x

    def get_init_x_random_single(self):

        abnormal_nodes = []

        mean = np.mean(self.features) # note, just for univariate feature
        std = np.std(self.features)
        for node in range(self.num_nodes):
            if self.features[node] >= mean + 2. * std:
                abnormal_nodes.append(node)

        init_x = np.zeros(self.num_nodes, dtype=np.float64)
        init_x[np.random.choice(abnormal_nodes)] = 1.

        return init_x

    def get_init_x_maxcc(self):
        """
        initialize x with max connected component
        :return:
        """

        abnormal_nodes = []
        mean = np.mean(self.features)
        std = np.std(self.features)
        for node in range(self.num_nodes):
            if self.features[node] >= mean + 2. * std:
                abnormal_nodes.append(node)

        abnormal_subgraph = nx.subgraph(self.graph, abnormal_nodes)
        # find the largest connected component in abnormal nodes
        largest_connected_component = max(nx.connected_component_subgraphs(abnormal_subgraph), key=len)

        init_x = np.zeros(self.num_nodes)
        init_x[largest_connected_component.nodes()] = 1.

        return init_x

    def get_init_x_zeros(self):

        return np.ones(self.num_nodes, dtype=np.float64)

    def get_gradient(self, x):

        sum_x = np.sum(x)
        if 0. == sum_x:
            print('gradient of x: entries of input vector x are all zeros !', file=sys.stderr)

        ct_x = np.dot(self.features, x)
        grad = self.features / np.sqrt(sum_x) - .5 * ct_x / np.power(sum_x, 1.5)

        if np.isnan(grad).any():
            print('something is wrong in gradient of x !', file=sys.stderr)

        return grad