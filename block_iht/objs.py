#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : objs
# @Date     : 03/30/2019 18:22:03
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


class GlobalEMS(object):

    def __init__(self, features, trade_off, bound=5, epsilon=1e-3, verbose=True):
        self.features = features
        self.num_nodes = len(features[0])
        self.num_time_stamps = len(features)
        self.trade_off = trade_off
        # self.max_iter = max_iter
        # self.learning_rate = learning_rate
        self.bound = bound
        self.epsilon = epsilon
        self.verbose = verbose

    def get_ems_val(self, x, features):

        sum_x = np.sum(x)
        ems_val = 0.

        if not 0. == sum_x:
            ct_x = np.dot(features, x)
            ems_val = ct_x / np.sqrt(sum_x)

        return ems_val

    def get_global_ems(self, x_array):
        x_vec = x_array.flatten()
        feature_vec = self.features.flatten()

        global_ems_val = self.get_ems_val(x_vec, feature_vec)

        return global_ems_val

    def get_obj_val(self, x_array):
        global_ems_val = self.get_global_ems(x_array)

        penalty = 0.
        for t in range(1, self.num_time_stamps):
            diff = x_array[t] - x_array[t - 1]
            penalty += np.linalg.norm(diff) ** 2

        obj_val = global_ems_val - self.trade_off * penalty

        return obj_val, global_ems_val, penalty

    def get_init_x_random(self):
        x_list = []
        for t in range(self.num_time_stamps):
            xt = np.random.rand(self.num_nodes)
            xt = np.array(xt >= 0.5, dtype=np.float64)
            x_list.append(xt)

        return np.array(x_list)

    def get_init_x_zeros(self):
        x_list = []
        for t in range(self.num_time_stamps):
            xt = np.zeros(self.num_nodes, dtype=np.float64)
            x_list.append(xt)

        return np.array(x_list)

    # def argmax_obj_with_proj(self, x_array, omega_x_list):
    #
    #     current_x_array = np.copy(x_array)
    #     for i in range(self.max_iter):
    #         prev_x_array = np.copy(current_x_array)
    #         for t in range(self.num_time_stamps):
    #             indicator_x = np.zeros_like(x_array[t])
    #             omega_x = omega_x_list[t]
    #             indicator_x[list(omega_x)] = 1.
    #             current_x = current_x_array[t]
    #             grad_x = self.get_gradient(current_x_array, t)


    def get_gradient(self, x_array, t):
        feature_vec = self.features[t]

        sum_x_array = np.sum(x_array)

        if 0. == sum_x_array:
            print('gradient_x, input x vector values are all zeros !', file=sys.stderr)
            grad = [0.] * self.num_nodes
        else:
            sum_ct_x_array = np.dot(self.features.flatten(), x_array.flatten())

            grad = feature_vec / np.sqrt(sum_x_array) - .5 * sum_ct_x_array / np.power(sum_x_array, 1.5)

        if 0 == t:
            grad += 2 * self.trade_off * (x_array[t+1] - x_array[t])
        elif self.num_time_stamps - 1 == t:
            grad -= 2 * self.trade_off * (x_array[t] - x_array[t-1])
        else:
            grad -= 2 * self.trade_off * (x_array[t] - x_array[t-1])
            grad += 2 * self.trade_off * (x_array[t+1] - x_array[t])

        if np.isnan(grad).any():
            raise ('something is wrong in gradient of x !')

        return grad

    # def _update_maximizer(self, grad_x, indicator_x, x):
    #
    #     updated_proj_x = (x + self.learning_rate * grad_x) * indicator_x
    #     sorted_indices = np.argsort(updated_proj_x)[::-1]
    #
    #     updated_proj_x[updated_proj_x < 0.] = 0.
    #     num_zero_psi = len(np.where(updated_proj_x == 0.))
    #     updated_proj_x[updated_proj_x > 1.] = 1.
    #     if num_zero_psi == len(x):
    #         print('siga-1 is too large and all values in the gradient are non-positive', file=sys.stderr)
    #         # select the first bound largest entries and set them 1s
    #         for i in range(self.bound):
    #             updated_proj_x[sorted_indices[i]] = 1.
    #
    #     return updated_proj_x


class BlockSumEMS(object):

    def __init__(self, features, trade_off, nodes_set=None, boundary_edges_dict=None, nodes_id_dict=None, verbose=True):
        self.features = features
        self.num_nodes = len(features)
        self.num_blocks = len(nodes_set) if nodes_set is not None else 0
        self.trade_off = trade_off
        self.nodes_set = nodes_set
        self.block_boundary_edges_dict = boundary_edges_dict
        self.node_id_dict = nodes_id_dict

        self.verbose = verbose

    def get_ems_value(self, x, feature_vector):
        sum_x = np.sum(x)

        if sum_x == 0.:
            return 0.
        else:
            ct_x = np.dot(feature_vector, x)
            func_val = ct_x / np.sqrt(sum_x)

        return func_val

    def get_sum_ems(self, x_array):
        score = 0.

        for b in range(self.num_blocks):
            x = x_array[sorted(self.nodes_set[b])]
            feat = self.features[sorted(self.nodes_set[b])]
            score += self.get_ems_value(x, feat)

        return score

    def get_obj_val(self, x_array, boundary_edges_dict):
        obj_val, sum_ems, penalty, smooth = 0., 0., 0., 0.
        sum_ems = self.get_sum_ems(x_array)

        for key in boundary_edges_dict:
            for (u, v) in boundary_edges_dict[key]:
                diff = x_array[u] - x_array[v]
                norm = np.linalg.norm(diff) ** 2
                smooth += norm
                penalty -= self.trade_off * norm
        obj_val = sum_ems + penalty

        return obj_val, sum_ems, penalty, smooth


    def get_init_x_random(self):
        x = np.random.rand(self.num_nodes)
        x = np.array(x >= 0.5, dtype=np.float64)

        return x

    def get_init_x_zeros(self):
        x = np.zeros(self.num_nodes, dtype=np.float64)

        return x


    def get_ems_grad(self, x, features):
        sum_x = np.sum(x)
        if 0. == sum_x:
            print('gradient_x: input x vector values are all zeros !!!', file=sys.stderr)
            ems_grad = [0.] * self.num_nodes
        else:
            ct_x = np.dot(features, x)
            ems_grad = features / np.sqrt(sum_x) - .5 * ct_x / np.power(sum_x, 1.5)

        if np.isnan(ems_grad).any():
            raise ('something is wrong in gradient of x.')

        return np.array(ems_grad)

    def get_penalty_grad(self, x, boundary_edge_x_dict):
        penalty_grad = [0.] * len(x)
        for (node_1, node_2) in boundary_edge_x_dict:
            adj_node_x = boundary_edge_x_dict[(node_1, node_2)]
            penalty_grad[node_1] += 2 * (x[node_1] - adj_node_x)

        if np.isnan(penalty_grad).any():
            raise ('something is wrong in gradient of x.')

        return np.array(penalty_grad)

    def get_gradient(self, x, features, boundary_edge_x_dict):
        grad = self.get_ems_grad(x, features)
        grad -= self.trade_off * self.get_penalty_grad(x, boundary_edge_x_dict)

        if np.isnan(grad).any():
            raise ('something is wrong in gradient of x.')

        return grad


class BlockGlobalEMS(object):

    def __init__(self, features, trade_off, nodes_set=None, boundary_edges_dict=None, nodes_id_dict=None, verbose=True):
        self.features = features
        self.num_nodes = len(features)
        self.num_blocks = len(nodes_set) if nodes_set is not None else 0
        self.trade_off = trade_off
        self.nodes_set = nodes_set
        self.block_boundary_edges_dict = boundary_edges_dict
        self.node_id_dict = nodes_id_dict

        self.verbose = verbose

    def get_ems_value(self, x, feature_vector):
        sum_x = np.sum(x)

        if sum_x == 0.:
            return 0.
        else:
            ct_x = np.dot(feature_vector, x)
            func_val = ct_x / np.sqrt(sum_x)

        return func_val

    def get_sum_ems(self, x_array):
        score = 0.

        for b in range(self.num_blocks):
            x = x_array[sorted(self.nodes_set[b])]
            feat = self.features[sorted(self.nodes_set[b])]
            score += self.get_ems_value(x, feat)

        return score

    def get_obj_val(self, x_array, boundary_edges_dict):
        obj_val, sum_ems, penalty, smooth = 0., 0., 0., 0.
        sum_ems = self.get_sum_ems(x_array)

        for key in boundary_edges_dict:
            for (u, v) in boundary_edges_dict[key]:
                diff = x_array[u] - x_array[v]
                norm = np.linalg.norm(diff) ** 2
                smooth += norm
                penalty -= self.trade_off * norm
        obj_val = sum_ems + penalty

        return obj_val, sum_ems, penalty, smooth


    def get_init_x_random(self):
        x = np.random.rand(self.num_nodes)
        x = np.array(x >= 0.5, dtype=np.float64)

        return x

    def get_init_x_zeros(self):
        x = np.zeros(self.num_nodes, dtype=np.float64)

        return x


    def get_ems_grad(self, x, features):
        sum_x = np.sum(x)
        if 0. == sum_x:
            print('gradient_x: input x vector values are all zeros !!!', file=sys.stderr)
            ems_grad = [0.] * self.num_nodes
        else:
            ct_x = np.dot(features, x)
            ems_grad = features / np.sqrt(sum_x) - .5 * ct_x / np.power(sum_x, 1.5)

        if np.isnan(ems_grad).any():
            raise ('something is wrong in gradient of x.')

        return np.array(ems_grad)

    def get_penalty_grad(self, x, boundary_edge_x_dict):
        penalty_grad = [0.] * len(x)
        for (node_1, node_2) in boundary_edge_x_dict:
            adj_node_x = boundary_edge_x_dict[(node_1, node_2)]
            penalty_grad[node_1] += 2 * (x[node_1] - adj_node_x)

        if np.isnan(penalty_grad).any():
            raise ('something is wrong in gradient of x.')

        return np.array(penalty_grad)

    def get_gradient(self, x, features, boundary_edge_x_dict):
        grad = self.get_ems_grad(x, features)
        grad -= self.trade_off * self.get_penalty_grad(x, boundary_edge_x_dict)

        if np.isnan(grad).any():
            raise ('something is wrong in gradient of x.')

        return grad
