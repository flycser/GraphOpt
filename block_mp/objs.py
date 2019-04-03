#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : objs
# @Date     : 03/30/2019 18:21:54
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


from __future__ import print_function

import sys

import numpy as np


class GlobalEMS(object):

    def __init__(self, features, trade_off, max_iter=1000, learning_rate=0.001, bound=5, epsilon=1e-3):
        self.features = features
        self.num_nodes = len(features[0])
        self.num_time_stamps = len(features)
        self.trade_off = trade_off
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.bound = bound
        self.epsilon = epsilon

    def get_ems_val(self, x, features):

        sum_x= np.sum(x)
        ems_val = 0.

        if not 0. == sum_x:
            ct_x = np.dot(features, x)
            ems_val = ct_x / sum_x

        return ems_val


    def get_global_ems(self, x_array):
        x_vec = []
        feature_vec = []
        for t in range(self.num_time_stamps):
            x_vec += list(x_array[t])
            feature_vec += self.features[t]

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
            xt = np.random.rand(self.num_nodes, dtype=np.float64)
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

            diff_norm_x_array = np.linalg.norm(current_x_array - prev_x_array)
            if diff_norm_x_array <= self.epsilon:
                break

        return current_x_array


    def get_gradient(self, x_array, t):

        feature_vec = self.features[t]

        sum_x_array = 0.
        for x in x_array:
            sum_x_array += np.sum(x)

        sum_ct_x_array = 0.
        if 0. == sum_x_array:
            print('gradient_x, input x vector values are all zeros !', file=sys.stderr)
            grad = [0.] * self.num_nodes
        else:
            for i, x in enumerate(x_array):
                sum_ct_x_array += np.dot(self.features[t], x)
            grad = feature_vec / np.sqrt(sum_x_array) - .5 * sum_ct_x_array / np.power(sum_x_array, 1.5)

        if 0 == t:
            grad += 2 * self.trade_off * (x_array[t+1] - x_array[t])
        elif self.num_time_stamps - 1 == t:
            grad - 2 * self.trade_off * (x_array[t] - x_array[t-1])
        else:
            grad -= 2 * self.trade_off * (x_array[t] - x_array[t-1])
            grad += 2 * self.trade_off * (x_array[t+1] - x_array[t])

        if np.isnan(grad).any():
            raise ('something is wrong in gradient of x !')

        return grad


        return None


    def _update_maximizer(self, grad_x, indicator_x, x):

        updated_proj_x = (x + self.learning_rate * grad_x) * indicator_x
        sorted_indices = np.argsort(updated_proj_x)[::-1]

        updated_proj_x[updated_proj_x < 0.] = 0.
        num_zero_psi = len(np.where(updated_proj_x == 0.))
        updated_proj_x[updated_proj_x > 1.] = 1.
        if num_zero_psi == len(x):
            print('', file=sys.stderr)
            # select the first bound largest entries and set them 1s
            for i in range(self.bound):
                updated_proj_x[sorted_indices[i]] = 1.

        return updated_proj_x
