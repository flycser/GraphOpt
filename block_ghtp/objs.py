#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : objs
# @Date     : 03/29/2019 22:15:51
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
# logger = logging.getLogger('fei') # todo, use as parameters


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
            diff = x_array[t] - x_array[t-1]
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

            # if 0 == i % 100 and self.verbose:
            #     obj_val, global_ems_val, penalty = self.get_obj_val(current_x_array)
            #     logger.debug('obj value={:.5f}, global_ems_value={:.5f}, penalty={:.5f}'.format(obj_val, global_ems_val, penalty))

            diff_norm_x_array = np.linalg.norm(current_x_array - prev_x_array)
            if diff_norm_x_array <= self.epsilon:
                break

        return current_x_array


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
            grad += 2 * self.trade_off * (x_array[t + 1] - x_array[t])
        elif self.num_time_stamps - 1 == t:
            grad -= 2 * self.trade_off * (x_array[t] - x_array[t - 1])
        else:
            grad -= 2 * self.trade_off * (x_array[t] - x_array[t - 1])
            grad += 2 * self.trade_off * (x_array[t + 1] - x_array[t])

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
    def __init__(self, features, trade_off, nodes_set=None, boundary_edges_dict=None, nodes_id_dict=None, max_iter=2000, learning_rate=0.01, bound=5, epsilon=1e-3, verbose=True):
        self.features = features
        self.num_nodes = len(features)
        self.num_blocks = len(nodes_set) if nodes_set is not None else 0
        self.trade_off = trade_off
        self.nodes_set = nodes_set
        self.boundary_edges_dict = boundary_edges_dict
        self.nodes_id_dict = nodes_id_dict
        # self.current_X = None
        # self.current_Y = None
        # self.current_Z = None
        # self.Omega_X = None
        # self.current_Z_next = None

        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.bound = bound
        self.epsilon = epsilon
        self.verbose = verbose

    def get_ems_score(self, x, feature_vector):

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
            x = x_array[sorted(self.nodes_set[b])] # get x of current block
            feat = self.features[sorted(self.nodes_set[b])]
            score += self.get_ems_score(x, feat)

        return score

    def get_obj_val(self, x_array, boundary_edges_dict):
        obj_val, sum_ems, penalty, smooth = 0., 0., 0., 0.
        sum_ems = self.get_sum_ems(x_array) # get summation of ems on all blocks

        # penalty on edge smooth
        for key in boundary_edges_dict:
            for (u, v) in boundary_edges_dict[key]:
                diff = x_array[u] - x_array[v]
                norm = np.linalg.norm(diff) ** 2
                smooth += norm
                penalty -= self.trade_off * norm
        obj_val = sum_ems + penalty

        return obj_val, sum_ems, penalty, smooth

    def get_init_x_zeros(self):
        x = np.zeros(self.num_nodes, dtype=np.float64)

        return x

    def get_ems_grad(self, x, feat):
        """
        gradient of ems part on block x
        :param x:
        :param feat:
        :return:
        """

        sum_x = np.sum(x)

        if 0.0 == sum_x:
            print('gradient_x: input x vector values are all zeros !!!')
            # grad = [0.] * self.num_nodes # todo
            grad = np.zeros(len(x))
        else:
            ct_x = np.dot(feat, x)
            grad = feat / np.sqrt(sum_x) - .5 * ct_x / np.power(sum_x, 1.5)

        if np.isnan(grad).any():
            raise ('something is wrong in gradient of x.')

        return grad

    def get_penalty_grad(self, x, boundary_xs_dict):
        grad = [0.] * len(x)
        for (u, v) in boundary_xs_dict:  # u is the node_id_in_block, v is the adjacent node id in other blocks
            adj_node_x = boundary_xs_dict[(u, v)]
            grad[u] += 2 * self.trade_off * (x[u] - adj_node_x)

        if np.isnan(grad).any():
            raise ('something is wrong in gradient of x.')

        return grad

    def get_gradient(self, x, feat, boundary_xs_dict):
        grad = self.get_ems_grad(x, feat)
        grad -= self.get_penalty_grad(x, boundary_xs_dict)

        if np.isnan(grad).any():
            raise ('something is wrong in gradient of x.')

        return grad

    def get_boundary_xs(self, x_array, boundary_edges, nodes_id_dict):

        boundary_xs_dict = {}
        for (u, v) in boundary_edges:
            node_id_in_block = nodes_id_dict[u]
            adj_x_val = x_array[v]
            boundary_xs_dict[(node_id_in_block, v)] = adj_x_val

        return boundary_xs_dict


    def argmax_obj_with_proj_serial_acc(self, x_array, omega_x_list):
        current_x_array = np.copy(x_array)  # we dont't want to modify the XT
        # prev_x_array = np.copy(x_array)
        # x_hat_array = np.copy(x_array)
        # obj_val_list = []
        p = 1.
        for i in range(self.max_iter):
            prev_x_array = np.copy(current_x_array)
            # print("iteration {}, time: {}".format(i, time.asctime(time.localtime(time.time()))))
            for b in range(self.num_blocks):
                indicator_x = np.zeros_like(current_x_array[self.nodes_set[b]])
                omega_x = omega_x_list[b]
                indicator_x[list(omega_x)] = 1.  # used for projection
                current_x = current_x_array[sorted(self.nodes_set[b])] # x on current block
                feat = self.features[sorted(self.nodes_set[b])]
                boundary_xs_dict = self.get_boundary_xs(current_x_array, self.boundary_edges_dict[b], self.nodes_id_dict)  # key is boundary edge, value is adjacent x in other blocks
                grad_x = self.get_gradient(current_x, feat, boundary_xs_dict)
                p = (1 + np.sqrt(1 + 4 * p ** 2)) / 2.0
                w = (p - 1.0) / p
                current_x = self._update_maximizer_acc(grad_x, indicator_x, current_x, self.bound, self.learning_rate, w)
                current_x_array[sorted(self.nodes_set[b])] = current_x

            # if 0 == i % 100 and self.verbose:
            #     obj_val, sum_ems, penalty, smooth = self.get_obj_val(current_x_array, self.boundary_edges_dict)
                # logger.debug('obj value={:.5f}, sum_ems_value={:.5f}, penalty={:.5f}'.format(obj_val, sum_ems, penalty))

            diff_norm_x = np.linalg.norm(current_x_array - prev_x_array)
            if diff_norm_x <= self.epsilon :
                # print("i: {}, obj_val {}".format(i, self.get_obj_value(current_x_array, self.boundary_edges_dict)))
                break

        return current_x_array

    def _update_maximizer_acc(self, grad_x, indicator_x, x, bound, learning_rate, w):
        normalized_x = (x + w * learning_rate * grad_x) * indicator_x
        sorted_indices = np.argsort(normalized_x)[::-1] # descending order

        normalized_x[normalized_x <= 0.0] = 0.0
        num_non_psi = len(np.where(normalized_x == 0.0))
        normalized_x[normalized_x > 1.0] = 1.0
        if num_non_psi == len(x):
            print("siga-1 is too large and all values in the gradient are non-positive")
            for i in range(bound):
                normalized_x[sorted_indices[i]] = 1.0

        return normalized_x

class ParallelSumEMS(object):
    def __init__(self, features, trade_off, nodes_set=None, boundary_edges_dict=None, nodes_id_dict=None, max_iter=2000, learning_rate=0.001, bound=5, epsilon=1e-3, verbose=True, tao=20):
        self.features = features
        self.num_nodes = len(features)
        self.num_blocks = len(nodes_set) if nodes_set is not None else 0
        self.trade_off = trade_off
        self.nodes_set = nodes_set
        self.boundary_edges_dict = boundary_edges_dict
        self.nodes_id_dict = nodes_id_dict
        self.current_X = None
        self.current_Y = None
        self.current_Z = None
        self.Omega_X = None
        self.current_Z_next = None

        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.bound = bound
        self.epsilon = epsilon
        self.verbose = verbose
        self.tao = tao

    def get_ems_score(self, x, feature_vector):

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
            x = x_array[sorted(self.nodes_set[b])] # get x of current block
            feat = self.features[sorted(self.nodes_set[b])]
            score += self.get_ems_score(x, feat)

        return score

    def get_obj_val(self, x_array, boundary_edges_dict):
        obj_val, sum_ems, penalty, smooth = 0., 0., 0., 0.
        sum_ems = self.get_sum_ems(x_array) # get summation of ems on all blocks

        # penalty on edge smooth
        for key in boundary_edges_dict:
            for (u, v) in boundary_edges_dict[key]:
                diff = x_array[u] - x_array[v]
                norm = np.linalg.norm(diff) ** 2
                smooth +=  norm
                penalty -= self.trade_off * norm
        obj_val = sum_ems + penalty

        return obj_val, sum_ems, penalty, smooth

    def get_init_x_zeros(self):
        x = np.zeros(self.num_nodes, dtype=np.float64)

        return x

    def get_ems_grad(self, x, feat):
        sum_x = np.sum(x)

        if 0.0 == sum_x:
            print('gradient_x: input x vector values are all zeros !!!')
            grad = np.zeros(len(x))
        else:
            ct_x = np.dot(feat, x)
            grad = feat / np.sqrt(sum_x) - .5 * ct_x / np.power(sum_x, 1.5)

        if np.isnan(grad).any():
            raise ('something is wrong in gradient of x.')

        return grad

    def get_penalty_grad(self, x, boundary_xs_dict):
        grad = [0.] * len(x)
        for (u, v) in boundary_xs_dict:  # u is the node_id_in_block, v is the adjacent node id in other blocks
            adj_node_x = boundary_xs_dict[(u, v)]
            grad[u] += 2 * self.trade_off * (x[u] - adj_node_x)

        if np.isnan(grad).any():
            raise ('something is wrong in gradient of x.')

        return grad

    def get_gradient(self, x, feat, boundary_xs_dict):
        grad = self.get_ems_grad(x, feat)
        grad -= self.get_penalty_grad(x, boundary_xs_dict)

        if np.isnan(grad).any():
            raise ('something is wrong in gradient of x.')

        return grad

    def get_boundary_xs(self, x_array, boundary_edges, nodes_id_dict):

        boundary_xs_dict = {}
        for (u, v) in boundary_edges:
            node_id_in_block = nodes_id_dict[u]
            adj_x_val = x_array[v]
            boundary_xs_dict[(node_id_in_block, v)] = adj_x_val

        return boundary_xs_dict


    def argmax_obj_with_proj_parallel(self, x_array, omega_x_list):
        tao = self.tao # number of cpus
        # tao_2 = 50 if self.num_blocks / 2 > 50 else self.num_blocks / 2 # note, number of cpus
        n = self.num_blocks
        theta = tao / float(n)
        # theta = tao_2 / float(n)

        # compute gradient for each x
        current_x_array = np.copy(x_array)  # we dont't want to modify the x_array
        current_z_array = np.copy(current_x_array)
        v = 1.0

        for i in range(self.max_iter):
            # print("iteration {}, time: {}".format(i, time.asctime(time.localtime(time.time()))))
            prev_x_array = np.copy(current_x_array)
            # convex combination of current_x_array and current_z_array
            current_y_array = (1 - theta) * current_x_array + theta * current_z_array
            # sampling some block to be updated
            block_samples = random.sample(range(self.num_blocks), tao)
            next_z_array = np.copy(current_z_array)

            paras_list = []
            manager = Manager()
            return_dict = manager.dict()
            jobs = []
            for block_index in block_samples:
                nodes = self.nodes_set[block_index]
                feat = self.features[sorted(nodes)]
                indicator_x = np.zeros_like(current_x_array[nodes])
                omega_x = omega_x_list[block_index]
                indicator_x[list(omega_x)] = 1.0  # used for projection
                boundary_xs_dict = self.get_boundary_xs(current_y_array, self.boundary_edges_dict[block_index], self.nodes_id_dict)  # key is boundary edge, value is adjacent x in other blocks
                current_y = current_y_array[nodes]
                next_z = next_z_array[nodes]
                current_x = current_x_array[nodes]
                paras = (block_index, nodes, feat, boundary_xs_dict, current_y, next_z, current_x, theta, tao, n, v, indicator_x)
                paras_list.append(paras)
                prs = Process(target=self.block_worker, args=(paras, return_dict))
                jobs.append(prs)
                prs.start()

            for proc in jobs:
                proc.join()

            for block_index in return_dict.keys():
                # print("block index ", block_index)
                next_z_array[self.nodes_set[block_index]] = return_dict[block_index]

            current_x_array = current_y_array + n / tao * theta * (next_z_array - current_z_array)
            theta = (np.sqrt(theta ** 4 + 4 * theta ** 2) - theta ** 2) / 2.0
            current_z_array = next_z_array

            diff_norm_x = np.linalg.norm(current_x_array - prev_x_array)
            if diff_norm_x <= self.epsilon:
                # print("iteration: {}, obj_val {}".format(i, self.get_obj_val(current_x_array, self.boundary_edges_dict)[0]))
                break

        return current_x_array

    def get_block_grad(self, z, z_next, feat, y, boundary_xs_dict, theta, tao, n, v):
        grad = self.get_ems_grad(z_next, feat)
        grad -= self.get_penalty_grad(y, boundary_xs_dict)

        grad -= float(n * theta * v) / tao * (z_next - z)

        return grad

    def block_worker(self, paras, return_dict):
        block_index, nodes, feat, boundary_xs_dict, y, z, x, theta, tao, n, v, indicator_x = paras

        z_next = np.copy(x)
        step_size = 0.001
        for k in range(2000):
            z_prev = np.copy(z_next)
            grad = self.get_block_grad(z, z_next, feat, y, boundary_xs_dict, theta, tao, n, v)
            z_next = self._update_maximizer_block(grad, indicator_x, z_next, bound=5, learning_rate=step_size)
            diff = np.linalg.norm(z_next - z_prev)
            if diff <= 1e-3:
                break

        return_dict[block_index] = z_next

    def _update_maximizer_block(self, grad_x, indicator_x, x, bound, learning_rate):

        normalized_x = (x + learning_rate * grad_x) * indicator_x
        sorted_indices = np.argsort(normalized_x)[::-1]

        normalized_x[normalized_x <= 0.0] = 0.0
        num_non_psi = len(np.where(normalized_x == 0.0))
        normalized_x[normalized_x > 1.0] = 1.0
        if num_non_psi == len(x):
            print("siga-1 is too large and all values in the gradient are non-positive")
            for i in range(bound):
                normalized_x[sorted_indices[i]] = 1.0

        return normalized_x


class BlockSumEMS(object):
    def __init__(self, features=None, num_blocks=None, nodes_set=None, boundary_edges_dict=None, nodes_id_dict=None, trade_off=10.):
        """
        In baojian's Java code, inputs are base and count, and base is used when solving argmax problem
        :param feature_vector:
        :param lambda_:
        """
        self.features = features
        self.num_nodes = len(features)
        self.num_blocks = num_blocks
        self.trade_off = trade_off
        self.nodes_set = nodes_set
        self.boundary_edges_dict = boundary_edges_dict
        self.nodes_id_dict = nodes_id_dict

    def get_boundary_xs(self, X, boundary_edges, nodes_id_dict):
        """get the boundary_xs_dict, which key is boundary edge with node_id in block relabeled,
        value is the x-val of adj node """
        boundary_xs_dict = {}
        for (u, v) in boundary_edges:
            node_id_in_block = nodes_id_dict[u]
            adj_x_val = X[v]
            boundary_xs_dict[(node_id_in_block, v)] = adj_x_val
        return boundary_xs_dict

    def get_ems_score(self, x, feature_vector):
        """
        get EMS value for current x (of each window)
        :param x:
        :return:
        """
        sum_x = np.sum(x)
        if sum_x == 0.:
            return 0.
        else:
            ct_x = np.dot(feature_vector, x)
            func_val = ct_x / np.sqrt(sum_x)  # function val should not consider regularization
        return func_val

    def get_sum_ems(self, X):
        score = 0.
        for t in range(self.num_blocks):
            xt = X[sorted(self.nodes_set[t])]
            fea = self.features[sorted(self.nodes_set[t])]
            score += self.get_ems_score(xt, fea)
        return score

    def get_obj_val(self, X, boundary_edges_dict):
        score, score1, score2, score3 = 0., 0., 0., 0.
        score1 = self.get_sum_ems(X)

        for key in boundary_edges_dict:
            for (u, v) in boundary_edges_dict[key]:
                diff = X[u] - X[v]
                score2 -= self.trade_off * (np.linalg.norm(diff) ** 2)
                score3 -= np.linalg.norm(diff) ** 2
        score = score1 + score2
        return score, score1, score2, score3

    def get_init_point_random(self):
        """
        create a random binary vector as initial vector for each x
        :return:
        """
        X = np.zeros(self.num_nodes, dtype=np.float64) + 0.000001
        return X

    def get_init_point_zeros(self):
        """
        create a random binary vector as initial vector for each x
        :return:
        """
        X = np.zeros(self.num_nodes, dtype=np.float64)
        return X

    def get_ems_grad(self, xt, fea):
        """
        maximize objective function (the sum of EMS + penalty), note the correctness of the format of gradient
        :param xt: current xt
        :param boundary_xs_dict: key is relabeled edge (node_id_in_block, v), and value is the x-value of adjacent node in other blocks
        :return:
        """
        sum_x = np.sum(xt)
        if 0.0 == sum_x:
            print('gradient_x: input x vector values are all zeros !!!')
            # grad = [0.] * self.num_nodes
            grad = np.zeros(self.num_nodes)
        else:
            ct_x = np.dot(fea, xt)
            # print(sum_x)
            grad = fea / np.sqrt(sum_x) - .5 * ct_x / np.power(sum_x, 1.5)

        if np.isnan(grad).any():
            raise ('something is wrong in gradient of x.')
        return grad

    def get_penality_grad(self, xt, boundary_xs_dict):
        grad = [0.] * len(xt)
        for (u, v) in boundary_xs_dict:  # u is the node_id_in_block, v is the adjacent node id in other blocks
            adj_node_x = boundary_xs_dict[(u, v)]
            grad[u] += 2 * self.trade_off * (xt[u] - adj_node_x)

        if np.isnan(grad).any():
            raise ('something is wrong in gradient of x.')
        return grad

    def get_gradient(self, xt, fea, boundary_xs_dict):
        grad = self.get_ems_grad(xt, fea)
        grad -= self.get_penality_grad(xt, boundary_xs_dict)

        if np.isnan(grad).any():
            raise ('something is wrong in gradient of x.')
        return grad


    def get_argmax_fx_with_proj_accelerated(self, XT, Omega_X, iter):
        """ closed form
        get argmax with projected gradient descent
        :param XT: all points for all blocks
        :param Omega_XT: the feasible set
        :return:
        """
        # TODO: figure out those hyperparameters
        lr, max_iter, bound = 0.001, 2000, 5
        print("lr {} max_iter {} ".format(lr, max_iter))
        p = 1.0

        # compute gradient for each x
        X = np.copy(XT)  # we dont't want to modify the XT
        obj_val_list = []
        for i in range(max_iter):
            X_pre = np.copy(X)
            # print("iteration {}, time: {}".format(i, time.asctime(time.localtime(time.time()))))
            for t in range(self.num_blocks):
                index = i * self.num_blocks + t
                indicator_x = np.zeros_like(X[self.nodes_set[t]])
                omega_x = Omega_X[t]
                indicator_x[list(omega_x)] = 1.0  # used for projection
                xt = X[sorted(self.nodes_set[t])]
                fea = self.features[sorted(self.nodes_set[t])]
                boundary_xs_dict = self.get_boundary_xs(X, self.boundary_edges_dict[t], self.nodes_id_dict)  # key is boundary edge, value is adjacent x in other blocks
                grad_xt = self.get_gradient(xt, fea, boundary_xs_dict)
                p = (1 + np.sqrt(1 + 4 * p ** 2)) / 2.0
                w = (p - 1.0) / p
                xt = self._update_maximizer_accelerated(grad_xt, indicator_x, xt, bound, lr, w)
                # print("t={}, {}".format(t, sorted(np.nonzero(xt)[0])))
                X[sorted(self.nodes_set[t])] = xt

            # obj_val =  self.get_obj_value(X, self.boundary_edges_dict)[0]
            # print("i: {}, obj_val {}, current lr {}".format(i, obj_val, lr))
            # if len(obj_val_list) > 50 and iter == 0:
            #     if obj_val < obj_val_list[-1]:
            #        lr = lr * 0.5
            #     elif (np.linalg.norm(obj_val - obj_val_list[-1]) ** 2) < 1e-4:
            #         lr = lr * 0.5
            #     # elif (np.linalg.norm(obj_val - obj_val_list[-1]) ** 2) < 1e-6:
            #     #     break
            #     obj_val_list.append(obj_val)
            # elif len(obj_val_list) >= 1 and iter != 0:
            #     if obj_val < obj_val_list[-1]:
            #         lr = lr * 0.5
            #     elif (np.linalg.norm(obj_val - obj_val_list[-1]) ** 2) < 1e-4:
            #         lr = lr * 0.5
            #     # elif (np.linalg.norm(obj_val - obj_val_list[-1]) ** 2) < 1e-6:
            #     #     break
            #     obj_val_list.append(obj_val)
            # else:
            #     obj_val_list.append(obj_val)

            diff_norm_x = np.linalg.norm(X - X_pre) ** 2
            if diff_norm_x <= 1e-6:
                break
        return X

    def _update_maximizer_accelerated(self, grad_x, indicator_x, x, bound, step_size, w):
        # normalized_x = (x + step_size * grad_x) * indicator_x  # update and project
        normalized_x = (x + w * step_size * grad_x) * indicator_x
        sorted_indices = np.argsort(normalized_x)[::-1]

        normalized_x[normalized_x <= 0.0] = 0.0
        num_non_posi = len(np.where(normalized_x == 0.0))
        normalized_x[normalized_x > 1.0] = 1.0
        if num_non_posi == len(x):
            print("siga-1 is too large and all values in the gradient are non-positive")
            for i in range(bound):
                normalized_x[sorted_indices[i]] = 1.0
        return normalized_x

    # ------------------------------------------------------------------------------------------------------------
    def get_argmax_fx_with_proj_accelerated_2(self, XT, Omega_X):
        """ closed form
        get argmax with projected gradient descent
        :param XT: all points for all blocks
        :param Omega_XT: the feasible set
        :return:
        """
        # TODO: figure out those hyperparameters
        lr, max_iter, bound = 0.001, 2000, 5
        p = 1.0

        # compute gradient for each x
        X = np.copy(XT)  # we dont't want to modify the XT
        X_prev = np.copy(XT)
        X_hat = np.copy(XT)
        obj_val_list = []
        for i in range(max_iter):
            print("iteration {}, time: {}".format(i, time.asctime(time.localtime(time.time()))))
            # X_prev = np.copy(X)
            for t in range(self.num_blocks):
                indicator_x = np.zeros_like(X[self.nodes_set[t]])
                omega_x = Omega_X[t]
                indicator_x[list(omega_x)] = 1.0  # used for projection
                xt = X[sorted(self.nodes_set[t])]
                xt_prev = X_prev[sorted(self.nodes_set[t])]
                p = (1+ np.sqrt(1+4*p**2)) / 2.0
                w = (p-1.0)/p
                xt_hat = xt + w * (xt - xt_prev) # fixme, add or minus
                X_hat[sorted(self.nodes_set[t])] = xt_hat
                X_prev[sorted(self.nodes_set[t])] = xt
                fea = self.features[sorted(self.nodes_set[t])]
                boundary_xs_dict = self.get_boundary_xs(X_hat, self.boundary_edges_dict[t], self.nodes_id_dict)  # key is boundary edge, value is adjacent x in other blocks
                xt = self.grad_ascent(xt_prev, xt_hat, fea, boundary_xs_dict, indicator_x)

                # fea = self.features[sorted(self.nodes_set[t])]
                # boundary_xs_dict = self.get_boundary_xs(X, self.boundary_edges_dict[t], self.nodes_id_dict)  # key is boundary edge, value is adjacent x in other blocks
                # xt = self.grad_ascent(xt, xt, fea, boundary_xs_dict, indicator_x)
                X[sorted(self.nodes_set[t])] = xt

            obj_val = self.get_obj_val(X, self.boundary_edges_dict)
            if i >= 1 and obj_val < obj_val_list[-1]:
                print("i: {}, obj_val {}".format(i, self.get_obj_val(X, self.boundary_edges_dict)))
                break
            obj_val_list.append(obj_val)
            # print("i: {}, obj_val {}".format(i, self.get_obj_value(X, self.boundary_edges_dict)))
            diff_norm_x = np.linalg.norm(X - X_prev) ** 2
            if i >= 1 and diff_norm_x <= 1e-6:
                print("i: {}, obj_val {}".format(i, self.get_obj_val(X, self.boundary_edges_dict)))
                break
        return X

    def get_block_grad(self, xt_hat, x_next, fea, boundary_xs_dict, step_size):

        # # fixme
        # if np.all(fea == 0.):
        #     fea += 1e-6

        grad = self.get_ems_grad(xt_hat, fea)
        grad -= self.get_penality_grad(xt_hat, boundary_xs_dict)
        grad -= 1.0 / step_size * (x_next - xt_hat)
        if np.isnan(grad).any():
            raise ('something is wrong in gradient of x.')
        return grad

    def grad_ascent(self, xt, xt_hat, fea, boundary_xs_dict, indicator_x):
        # x_next = np.copy(xt)
        x_next = np.zeros(len(xt)) + 1e-6
        step_size = 0.001
        for k in range(5000):
            x_prev = np.copy(x_next)
            grad_x_next = self.get_block_grad(xt_hat, x_next, fea, boundary_xs_dict, step_size)
            # print("grad_x_next {}".format(grad_x_next))
            x_next = self._update_maximizer_accelerated_2(grad_x_next, indicator_x, x_next, bound=5, step_size=step_size)
            diff = np.linalg.norm(x_next - x_prev) ** 2
            # print("diff {}".format(diff))
            if diff <= 1e-6:
                # print("k {}".format(k))
                break
        return x_next

    def _update_maximizer_accelerated_2(self, grad_x, indicator_x, x, bound, step_size):
        # normalized_x = (x + step_size * grad_x) * indicator_x  # update and project
        normalized_x = (x + step_size * grad_x) * indicator_x
        sorted_indices = np.argsort(normalized_x)[::-1]

        normalized_x[normalized_x <= 0.0] = 0.0
        num_non_posi = len(np.where(normalized_x == 0.0))
        normalized_x[normalized_x > 1.0] = 1.0
        if num_non_posi == len(x):
            print("siga-1 is too large and all values in the gradient are non-positive")
            for i in range(bound):
                normalized_x[sorted_indices[i]] = 1.0
        return normalized_x


class DualEMS(object):

    def __init__(self, features, trade_off, max_iter=2000, learning_rate=0.001, bound=5, epsilon=1e-3, verbose=True):

        self.features = features
        self.num_nodes = len(features)

        self.trade_off = trade_off
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.bound = bound
        self.epsilon = epsilon

        self.verbose = verbose

    def get_ems_val(self, x, feat_vec):
        sum_x = np.sum(x)
        ems_val = 0.

        if not 0. == sum_x:
            ct_x = np.dot(feat_vec, x)
            ems_val = ct_x / np.sqrt(sum_x)

        return ems_val

    def get_obj_val(self, x, y):
        x_ems_val = self.get_ems_val(x, self.features)
        y_ems_val = self.get_ems_val(y, self.features)

        penalty = 0.
        penalty += np.linalg.norm(x - y)

        obj_val = x_ems_val + y_ems_val - self.trade_off * penalty

        return obj_val, x_ems_val, y_ems_val, penalty


    def get_init_x_zeros(self):

        x = np.zeros(self.num_nodes, dtype=np.float64)
        y = np.zeros(self.num_nodes, dtype=np.float64)

        return x, y


    def get_gradient(self, x, y):

        sum_x = np.sum(x)
        if 0. == sum_x:
            print('gradient, input vector values are all zeros !', file=sys.stderr)
            grad_x = [0.] * self.num_nodes
        else:
            sum_ct_x = np.dot(self.features, x)
            grad_x = self.features / np.sqrt(sum_x) - .5 * sum_ct_x / np.power(sum_x, 1.5) # gradient for one graph

        # penalty
        grad_x -= 2 * self.trade_off * (x - y)

        if np.isnan(grad_x).any():
            raise ('something is wrong in gradient of x !')

        return grad_x


    def argmax_obj_with_proj(self, x, y, omega_x, omega_y):

        current_x, current_y = np.copy(x), np.copy(y)
        for i in range(self.max_iter):
            prev_x, prev_y = np.copy(current_x), np.copy(current_y)

            # update first graph
            indicator_x = np.zeros_like(current_x)
            indicator_x[list(omega_x)] = 1.
            grad_x = self.get_gradient(current_x, current_y)

            current_x = self._update_maximizer(grad_x, indicator_x, current_x)

            # update second graph
            indicator_y = np.zeros_like(current_y)
            indicator_y[list(omega_x)] = 1.
            grad_y = self.get_gradient(current_y, current_x)

            current_y = self._update_maximizer(grad_y, indicator_y, current_y)


            diff_norm = np.sqrt(np.linalg.norm(current_x - prev_x) ** 2 + np.linalg.norm(current_y - prev_y) ** 2)
            if diff_norm <= self.epsilon:
                break

        return current_x, current_y

    def _update_maximizer(self, grad_x, indicator_x, x):
        # update and projection
        updated_proj_x = (x + self.learning_rate * grad_x) * indicator_x
        sorted_indices = np.argsort(updated_proj_x)[::-1]

        # restrict x in [0, 1]
        updated_proj_x[updated_proj_x < 0.] = 0.
        num_zero_psi = len(np.where(updated_proj_x == 0.))
        updated_proj_x[updated_proj_x > 1.] = 1.
        if num_zero_psi == len(x):
            print('siga-1 is too large and all values in the gradient are non-positive', file=sys.stderr)
            # select the first bound largest entries and set them 1s
            for i in range(self.bound):
                updated_proj_x[sorted_indices[i]] = 1.

        return updated_proj_x