#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : obj
# @Date     : 03/27/2019 14:06:18
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     : objective function classes


from __future__ import print_function

import sys

import numpy as np


class EMS(object):

    def __init__(self, features, max_iter=1000, learning_rate=0.001, bound=5, epsilon=1e-3):

        self.features = features
        self.num_nodes = len(features)
        self.max_iter = max_iter # maximal iterations
        self.learning_rate = learning_rate
        self.bound = bound # non zeros entries
        self.epsilon = epsilon # terminate condition

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

    def get_init_x_ones(self):

        return np.ones(self.num_nodes, dtype=np.float64)

    def get_init_x_zeros(self):

        return np.ones(self.num_nodes, dtype=np.float64)

    def argmax_obj_with_proj(self, init_x, omega_x):

        current_x = np.copy(init_x)
        indicator_x = np.zeros_like(init_x)
        indicator_x[list(omega_x)] = 1.

        for iter in range(self.max_iter):
            grad_x = self.get_gradient(current_x)
            prev_x = np.copy(current_x)
            current_x = self._update_maximizer(grad_x, indicator_x, current_x)

            diff_norm_x = np.linalg.norm(current_x - prev_x)
            if diff_norm_x <= self.epsilon:
                break

        return current_x

    def get_gradient(self, x):

        sum_x = np.sum(x)
        if 0. == sum_x:
            print('gradient of x: entries of input vector x are all zeros !', file=sys.stderr)

        ct_x = np.dot(self.features, x)
        grad = self.features / np.sqrt(sum_x) - .5 * ct_x / np.power(sum_x, 1.5)

        if np.isnan(grad).any():
            print('something is wrong in gradient of x !', file=sys.stderr)

        return grad

    def _update_maximizer(self, grad_x, indicator_x, x):
        """
        update x, constrain each of its entries into [0,1] and supp(x) in omega set
        :param grad_x:
        :param indicator_x:
        :param x:
        :param bound:
        :param learning_rate:
        :return:
        """

        updated_proj_x = (x + self.learning_rate * grad_x) * indicator_x

        sorted_indices = np.argsort(updated_proj_x)[::-1]

        # constrain x in [0, 1]
        updated_proj_x[updated_proj_x < 0.] = 0.
        num_zero_psi = len(np.where(updated_proj_x == 0.))
        updated_proj_x[updated_proj_x > 1.] = 1.

        # handle possible errors resulting from all zero x after projection
        if num_zero_psi == len(x):
            print('', file=sys.stderr)
            # select the first bound largest entries and set them 1s
            for i in range(self.bound):
                updated_proj_x[sorted_indices[i]] = 1.

        return updated_proj_x