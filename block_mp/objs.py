#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : objs
# @Date     : 03/30/2019 18:21:54
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

import time
import random

from multiprocessing import Pool, Process, Manager

import numpy as np


class GlobalEMS(object):

    def __init__(self, features, trade_off, max_iter=2000, learning_rate=0.001, bound=5, epsilon=1e-3, verbose=True):
        self.features = features
        self.num_nodes = len(features[0])
        self.num_time_stamps = len(features)
        self.trade_off = trade_off
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.bound = bound
        self.epsilon = epsilon
        self.verbose = verbose

    def get_ems_val(self, x, features):

        sum_x= np.sum(x)
        ems_val = 0.

        if not 0. == sum_x:
            ct_x = np.dot(features, x)
            ems_val = ct_x / np.sqrt(sum_x)

        return ems_val


    def get_global_ems(self, x_array):
        # x_vec = []
        # feature_vec = []
        # for t in range(self.num_time_stamps):
        #     x_vec += list(x_array[t])
        #     feature_vec += list(self.features[t])
        x_vec = x_array.flatten()
        feature_vec = self.features.flatten()

        global_ems_val = self.get_ems_val(x_vec, feature_vec)

        return global_ems_val

    def get_obj_val(self, x_array):

        global_ems_val = self.get_global_ems(x_array)
        # penalty term
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
            xt = np.zeros(self.num_nodes, dtype=np.float64) # note, add 1e-6 when using it
            x_list.append(xt)

        return np.array(x_list)

    def argmax_obj_with_proj(self, x_array, omega_x_list):

        current_x_array = np.copy(x_array)
        for i in range(self.max_iter):
            prev_x_array = np.copy(current_x_array)
            for t in range(self.num_time_stamps):
                indicator_x = np.zeros_like(x_array[t])
                omega_x = omega_x_list[t]
                indicator_x[list(omega_x)] = 1.
                current_x = current_x_array[t]
                grad_x = self.get_gradient(current_x_array, t)
                current_x = self._update_maximizer(grad_x, indicator_x, current_x)
                current_x_array[t] = current_x

            if 0 == i % 100 and self.verbose:
                obj_val, global_ems_val, penalty = self.get_obj_val(current_x_array)
                logger.debug('obj value={:.5f}, global_ems_value={:.5f}, penalty={:.5f}'.format(obj_val, global_ems_val, penalty))

            diff_norm_x_array = np.linalg.norm(current_x_array - prev_x_array)
            if diff_norm_x_array <= self.epsilon:
                break

        return current_x_array

    def get_gradient(self, x_array, t):

        feature_vec = self.features[t]

        # sum_x_array = 0.
        # for x in x_array:
        #     sum_x_array += np.sum(x)
        sum_x_array = np.sum(x_array)
        # print('sum_x_array {:.5f}'.format(sum_x_array))
        # print('sum_x_array_2 {:.5f}'.format(sum_x_array_2))

        if 0. == sum_x_array:
            print('gradient_x, input x vector values are    all zeros !', file=sys.stderr)
            grad = [0.] * self.num_nodes
        else:
            sum_ct_x_array = np.dot(self.features.flatten(), x_array.flatten())
            # for i, x in enumerate(x_array):
            #     sum_ct_x_array += np.dot(self.features[i], x)
            # print('sum_ct_x_array {:.5f}'.format(sum_ct_x_array))
            # print('sum_ct_x_array_2 {:.5f}'.format(sum_ct_x_array_2))

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


    def _update_maximizer(self, grad_x, indicator_x, x):

        updated_proj_x = (x + self.learning_rate * grad_x) * indicator_x
        sorted_indices = np.argsort(updated_proj_x)[::-1]

        updated_proj_x[updated_proj_x < 0.] = 0.
        num_zero_psi = len(np.where(updated_proj_x == 0.))
        updated_proj_x[updated_proj_x > 1.] = 1.
        if num_zero_psi == len(x):
            print('siga-1 is too large and all values in the gradient are non-positive', file=sys.stderr)
            # select the first bound largest entries and set them 1s
            for i in range(self.bound):
                updated_proj_x[sorted_indices[i]] = 1.

        return updated_proj_x


class SerialSumEMS(object):

    def __init__(self, features, block_node_sets, node_id_dict, block_boundary_edges_dict, trade_off, max_iter=2000, learning_rate=0.001, bound=5, epsilon=1e-3, verbose=True):

        self.features = features
        self.num_nodes = len(features)
        self.num_blocks = len(block_node_sets)
        self.trade_off = trade_off
        self.block_node_sets = block_node_sets
        self.node_id_dict = node_id_dict
        self.block_boundary_edges_dict = block_boundary_edges_dict
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.bound = bound
        self.epsilon = epsilon
        self.verbose = verbose


    def get_ems_value(self, x, features):
        sum_x = np.sum(x)
        ems_val = 0.

        if not 0. == sum_x:
            ct_x = np.dot(features, x)
            ems_val = ct_x / np.sqrt(sum_x)

        return ems_val

    def get_sum_ems(self, x):

        sum_ems_val = 0.
        for b in range(self.num_blocks):
            block_x = x[sorted(self.block_node_sets[b])] # coefficient x in current block
            block_features = self.features[sorted(self.block_node_sets[b])] # features in current block
            block_ems = self.get_ems_value(block_x, block_features)
            sum_ems_val += block_ems

        return sum_ems_val

    def get_obj_val(self, x):

        sum_ems_val = self.get_sum_ems(x)
        penalty = 0.
        for b in self.block_boundary_edges_dict:
            for edge in self.block_boundary_edges_dict[b]:
                node_1, node_2 = edge
                diff = x[node_1] - x[node_2]
                penalty += np.linalg.norm(diff) ** 2

        obj_val = sum_ems_val - self.trade_off * penalty

        return obj_val, sum_ems_val, penalty

    def get_init_x_random(self):
        init_x = np.random.rand(self.num_nodes)
        init_x = np.array(init_x >= 0.5, dtype=np.float64)

        return init_x

    def get_init_x_zeros(self):
        init_x = np.zeros(self.num_nodes, dtype=np.float64)

        return init_x

    def get_ems_grad(self, x, features):
        sum_x = np.sum(x)
        if 0. == sum_x:
            print('gradient_x: input x vector values are all zeros !!!')
            ems_grad = [0.] * self.num_nodes
        else:
            ct_x = np.dot(features, x)
            ems_grad = features / np.sqrt(sum_x) - .5 * ct_x / np.power(sum_x, 1.5)

        if np.isnan(ems_grad).any():
            raise ('something is wrong in gradient of x.')

        return np.array(ems_grad)

    def get_penalty_grad(self, x, boundary_edge_x_dict):
        penalty_grad = [0.] * len(x)
        for (node_1, node_2) in boundary_edge_x_dict:  # node_1 is the node_id_in_block, node_2 is the adjacent node id in other blocks
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

    def argmax_obj_with_proj_serial(self, x, omega_x_list):

        p = 1.
        current_x = np.copy(x)
        for i in range(self.max_iter):
            prev_x = np.copy(current_x)
            for b in range(self.num_blocks):
                block_indicator_x = np.zeros_like(current_x[self.block_node_sets[b]])
                block_omega_x = omega_x_list[b]
                block_indicator_x[list(block_omega_x)] = 1.
                block_x = current_x[sorted(self.block_node_sets[b])]
                block_features = self.features[sorted(self.block_node_sets[b])]
                boundary_edge_x_dict = self.get_boundary_edge_x_dict(x, self.block_boundary_edges_dict[b], self.node_id_dict) # todo, simplify
                block_grad = self.get_gradient(block_x, block_features, boundary_edge_x_dict)
                p = (1 + np.sqrt(1 + 4 * p ** 2)) / 2.
                w = (p - 1.) / p
                block_x = self._update_maximizer_accelerated(block_grad, block_indicator_x, block_x, w)
                current_x[sorted(self.block_node_sets[b])] = block_x

            if  0 == i % 10 and self.verbose:
                obj_val, global_ems_val, penalty = self.get_obj_val(current_x, self.block_boundary_edges_dict)
                logger.debug('internal iter {:d}, obj value={:.5f}, sum ems value={:.5f}, penalty={:.5f}'.format(i, obj_val, global_ems_val, penalty))

            diff_norm_x = np.linalg.norm(current_x - prev_x)
            if diff_norm_x <= self.epsilon:
                break

        return current_x


    def get_boundary_edge_x_dict(self, x, boundary_edges, node_id_dict):
        boundary_edge_x_dict = {}
        for edge in boundary_edges:
            node_1, node_2 = edge
            block_node_id = node_id_dict[node_1]
            adj_x_val = x[node_2]
            boundary_edge_x_dict[(block_node_id, node_2)] = adj_x_val

        return boundary_edge_x_dict


    def _update_maximizer(self, grad_x, indicator_x, x):
        updated_proj_x = (x + self.learning_rate * grad_x) * indicator_x
        sorted_indices = np.argsort(updated_proj_x)[::-1]

        updated_proj_x[updated_proj_x < 0.] = 0.
        num_zero_psi = len(np.where(updated_proj_x == 0.))
        updated_proj_x[updated_proj_x > 1.] = 1.
        if num_zero_psi == len(x):
            print('siga-1 is too large and all values in the gradient are non-positive', file=sys.stderr)
            # select the first bound largest entries and set them 1s
            for i in range(self.bound):
                updated_proj_x[sorted_indices[i]] = 1.

        return updated_proj_x

    def _update_maximizer_accelerated(self, grad_x, indicator_x, x, w):
        updated_proj_x = (x + w * self.learning_rate * grad_x) * indicator_x
        sorted_indices = np.argsort(updated_proj_x)[::-1]

        updated_proj_x[updated_proj_x < 0.] = 0.
        num_zero_psi = len(np.where(updated_proj_x == 0.))
        updated_proj_x[updated_proj_x > 1.] = 1.
        if num_zero_psi == len(x):
            print('siga-1 is too large and all values in the gradient are non-positive', file=sys.stderr)
            # select the first bound largest entries and set them 1s
            for i in range(self.bound):
                updated_proj_x[sorted_indices[i]] = 1.

        return updated_proj_x


class ParallelSumEMS(object):
    def __init__(self, features, block_node_sets, node_id_dict, block_boundary_edges_dict, trade_off, max_iter=2000, learning_rate=0.001, bound=5, epsilon=1e-3, verbose=True):

        self.features = features
        self.num_nodes = len(features)
        self.num_blocks = len(block_node_sets)
        self.trade_off = trade_off
        self.block_node_sets = block_node_sets
        self.node_id_dict = node_id_dict
        self.block_boundary_edges_dict = block_boundary_edges_dict
        self.max_iter = max_iter
        self.learning_rate = learning_rate
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

    def get_sum_ems(self, x):
        sum_ems_val = 0.
        for b in range(self.num_blocks):
            block_x = x[sorted(self.block_node_sets[b])] # get coefficients in current block
            block_features = self.features[sorted(self.block_node_sets[b])] # get features in current block
            block_ems = self.get_ems_val(block_x, block_features)
            sum_ems_val += block_ems # summation of ems value

        return sum_ems_val

    def get_obj_val(self, x, block_boundary_edges_dict):
        sum_ems_val = self.get_sum_ems(x)
        penalty = 0.
        for b in self.block_boundary_edges_dict:
            for edge in self.block_boundary_edges_dict[b]:
                node_1, node_2 = edge
                diff = x[node_1] - x[node_2]
                penalty += np.linalg.norm(diff) ** 2

        obj_val = sum_ems_val - self.trade_off * penalty

        return obj_val, sum_ems_val, penalty

    def get_init_x_random(self):
        init_x = np.random.rand(self.num_nodes)
        init_x = np.array(init_x >= 0.5, dtype=np.float64)

        return init_x

    def get_init_x_zeros(self):
        init_x = np.zeros(self.num_nodes, dtype=np.float64)

        return init_x

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


    def argmax_obj_with_proj_parallel(self, x, omega_x_list):

        tao = 20 # number of cpus
        n = self.num_blocks
        theta = tao / float(n)

        current_x = np.copy(x)
        current_z = np.copy(current_x)
        v = 1.

        for i in range(self.max_iter):
            print('iteration {:d}, time {}'.format(i, time.asctime(time.localtime(time.time()))))
            prev_x = np.copy(current_x)
            current_y = (1 - theta) * current_x + theta * current_z
            # note, select blocks without replacement
            # selected_blocks = np.random.choice(range(self.num_blocks), tao, replace=False) # select tao blocks which are to be updated
            selected_blocks = random.sample(range(self.num_blocks), tao) # select tao blocks which are to be updated
            next_z = np.copy(current_z)
            para_list = []
            manager = Manager()
            return_dict = manager.dict()
            jobs = []

            # parallel jobs
            for b in selected_blocks:
                block_nodes = sorted(self.block_node_sets[b])
                block_features = self.features[block_nodes]
                block_indicator_x = np.zeros_like(current_x[block_nodes])
                block_omega_x = omega_x_list[b]
                block_indicator_x[list(block_omega_x)] = 1.
                block_boundary_edge_x_dict = self.get_boundary_edge_x_dict(current_y, self.block_boundary_edges_dict[b], self.node_id_dict)
                block_y = current_y[block_nodes]
                block_z = next_z[block_nodes]
                block_x = current_x[block_nodes]
                para = (b, block_nodes, block_features, block_boundary_edge_x_dict, block_y, block_z, block_x, theta, tao, n, v, block_indicator_x)
                para_list.append(para)
                process = Process(target=self._update_block_worker, args=(para, return_dict))
                jobs.append(process)
                process.start()

            for proc in jobs:
                proc.join()

            for b in return_dict.keys():
                next_z[sorted(self.block_node_sets[b])] = return_dict[b]

            current_x = current_y + n / tao * theta * (next_z - current_z)
            theta = (np.sqrt(theta ** 4 + 4 * theta ** 2) - theta ** 2) / 2.
            current_z = next_z

            diff_norm_x = np.linalg.norm(current_x - prev_x)
            if diff_norm_x <= self.epsilon:
                break

        return current_x


    def get_boundary_edge_x_dict(self, x, boundary_edges, node_id_dict):

        boundary_edge_x_dict = {}
        for edge in boundary_edges:
            node_1, node_2 = edge
            block_node_id = node_id_dict[node_1]
            adj_x_val = x[node_2]
            boundary_edge_x_dict[(block_node_id, node_2)] = adj_x_val

        return boundary_edge_x_dict

    def _update_maximizer_block(self, grad, indicator_x, x):

        updated_proj_x = (x + self.learning_rate * grad) * indicator_x
        sorted_indices = np.argsort(updated_proj_x)[::-1]

        updated_proj_x[updated_proj_x < 0.] = 0.
        num_non_posi = len(np.where(updated_proj_x == 0.))
        updated_proj_x[updated_proj_x > 1.] = 1.
        if num_non_posi == len(x):
            print('siga-1 is too large and all values in the gradient are non-positive', file=sys.stderr)
            for i in range(self.bound):
                updated_proj_x[sorted_indices[i]] = 1.

        return updated_proj_x

    def _update_block_worker(self, para, return_dict):

        b, block_nodes, block_features, block_boundary_edge_x_dict, block_y, block_z, block_x, theta, tao, n, v, block_indicator_x = para
        next_block_z = np.copy(block_x)
        for i in range(self.max_iter):
            prev_block_z = np.copy(next_block_z)
            block_grad = self.get_block_gradient(block_z, next_block_z, block_features, block_y, block_boundary_edge_x_dict, theta, tao, n, v)
            next_block_z = self._update_maximizer_block(block_grad, block_indicator_x, next_block_z)

            diff_norm_x = np.linalg.norm(next_block_z - prev_block_z)
            if diff_norm_x <= self.epsilon:
                break

        return_dict[b] = next_block_z
            
    def get_block_gradient(self, z, next_z, features, y, boundary_edge_x_dict, theta, tao, n, v):
        grad = self.get_ems_grad(next_z, features)
        grad -= self.trade_off * self.get_penalty_grad(y, boundary_edge_x_dict)
        grad -= float(n * theta * v) / tao * (next_z - z)
        
        return grad
    
    