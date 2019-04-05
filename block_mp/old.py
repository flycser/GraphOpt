#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : old
# @Date     : 04/04/2019 17:31:33
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


from __future__ import print_function

import subprocess
import os
import time
from multiprocessing import Pool, Lock
from sparse_learning.proj_algo import head_proj
from sparse_learning.proj_algo import tail_proj

# from post_process import *
import networkx as nx
import numpy as np
import pickle
# import EMS

logger = True
output_lock = Lock()


def generate_combined_subgraphs(G, T, pred_subgraphs):
    """output the predicted subgraphs of multiple time-stamps as one graph, and give new ID to same nodes at different time stamp"""
    num_nodes = len(G.nodes())
    pred_sub = pred_subgraphs[0]
    S = G.subgraph(pred_sub)
    newid_oldid_dict = {}
    oldid_newid_dict = {0: {}}  # first key is the time-stamp, and the second key is the oldid
    new_id = -1
    edges = []
    feature_vec = []
    weights = []

    for node in S.nodes():
        if node not in oldid_newid_dict[0]:
            new_id += 1
            oldid_newid_dict[0][node] = new_id
            newid_oldid_dict[new_id] = (0, node)
    for (u, v) in S.edges():
        edges.append((oldid_newid_dict[0][u], oldid_newid_dict[0][v]))

    for t in range(1, T):
        if t not in oldid_newid_dict:
            oldid_newid_dict[t] = {}
        pred_sub = pred_subgraphs[t]
        overlap = set(pred_subgraphs[t]).intersection(pred_subgraphs[t - 1])
        S = G.subgraph(pred_sub)

        for node in S.nodes():
            if node not in oldid_newid_dict[t]:
                new_id += 1
                oldid_newid_dict[t][node] = new_id
                newid_oldid_dict[new_id] = (t, node)
        for (u, v) in S.edges():
            edges.append((oldid_newid_dict[t][u], oldid_newid_dict[t][v]))

        for node_id in overlap:
            current_id = oldid_newid_dict[t][node_id]
            prev_id = oldid_newid_dict[t - 1][node_id]
            edges.append((current_id, prev_id))
    # return np.array(edges), np.array(feature_vec), np.array(weights), newid_oldid_dict
    return np.array(edges), newid_oldid_dict


def refine_predicted_subgraph(largest_cc, newid_oldid_dict, T):
    refined_pred_subgraphs = [[] for i in range(T)]

    for node in largest_cc:
        t, old_id = newid_oldid_dict[node]
        refined_pred_subgraphs[t].append(old_id)
    print("refined_pred_subgraphs {}".format(refined_pred_subgraphs))
    return refined_pred_subgraphs


def evaluate(true_subgraphs, pred_subgraphs, log_file=None):
    T = len(true_subgraphs)
    true_subgraphs_size = 0.
    pred_subgraphs_size = 0.
    valid_pred_subgraphs_size = 0.
    valid_intersection = 0.
    valid_union = 0.
    all_intersection = 0.
    all_union = 0.
    for t in range(T):
        true_subgraph, pred_subgraph = set(list(true_subgraphs[t])), set(list(pred_subgraphs[t]))
        true_subgraphs_size += len(true_subgraph)
        pred_subgraphs_size += len(pred_subgraph)
        intersection = true_subgraph.intersection(pred_subgraph)
        union = true_subgraph.union(pred_subgraph)
        all_intersection += len(intersection)
        all_union += len(union)

        if len(true_subgraph) != 0:
            valid_pred_subgraphs_size += len(pred_subgraph)
            valid_intersection += len(intersection)
            valid_union += len(union)

    if pred_subgraphs_size != 0.:
        global_prec = all_intersection / float(pred_subgraphs_size)
    else:
        global_prec = 0.
    global_rec = all_intersection / float(true_subgraphs_size)
    if global_prec + global_rec != 0.:
        global_fm = (2. * global_prec * global_rec) / (global_prec + global_rec)
    else:
        global_fm = 0.
    global_iou = all_intersection / float(all_union)

    if valid_pred_subgraphs_size != 0.:
        valid_global_prec = valid_intersection / float(valid_pred_subgraphs_size)
    else:
        valid_global_prec = 0.
    valid_global_rec = valid_intersection / true_subgraphs_size
    if valid_global_prec + valid_global_rec != 0.:
        valid_global_fm = (2. * valid_global_prec * valid_global_rec) / (valid_global_prec + valid_global_rec)
    else:
        valid_global_fm = 0.
    valid_global_iou = valid_intersection / float(valid_union)

    return global_prec, global_rec, global_fm, global_iou, valid_global_prec, valid_global_rec, valid_global_fm, valid_global_iou

class GlobalEMS(object):
    def __init__(self, feature_matrix=None, trade_off=10.):
        """
        In baojian's Java code, inputs are base and count, and base is used when solving argmax problem
        :param feature_vector:
        :param lambda_:
        """
        self.feature_matrix = feature_matrix
        self.T = len(feature_matrix)
        self.num_nodes = len(feature_matrix[0])
        self.trade_off = trade_off

    def get_ems_score(self, x, feature_vector):
        """
        get EMS value for current x (of each window)
        :param x:
        :return:
        """
        sum_x = np.sum(x)
        # sum_x = len(np.nonzero(x))
        if sum_x == 0.:
            return 0.
        else:
            ct_x = np.dot(feature_vector, x)
            func_val = ct_x / np.sqrt(sum_x)  # function val should not consider regularization
        return func_val

    def get_global_ems(self, X):
        x = []
        fea = []
        for t in range(self.T):
            x += list(X[t])
            fea += list(self.feature_matrix[t])
        global_ems_score = self.get_ems_score(x, fea)
        return global_ems_score

    def get_obj_value(self, X):
        score, score1, score2, score3 = 0., 0., 0., 0.
        score1 = self.get_global_ems(X)
        for t in range(1, self.T):
            diff = X[t] - X[t - 1]
            score2 -= self.trade_off * (np.linalg.norm(diff) ** 2)
            score3 -= np.linalg.norm(diff) ** 2
        score = score1 + score2
        # print("penalty: ", score2)
        return score, score1, score2, score3

    def get_init_point_random(self):
        """
        create a random binary vector as initial vector for each x
        :return:
        """
        XT = []
        for t in range(self.T):
            # xt_0 = np.random.rand(self.num_nodes)
            # xt_0 = np.array(xt_0 >= 0.5, dtype=np.float64)
            # xt_0 = np.ones(self.num_nodes, dtype=np.float64)
            xt_0 = np.zeros(self.num_nodes, dtype=np.float64) + 0.000001
            XT.append(xt_0)
        return np.array(XT)
        # return np.ones((self.T, self.num_nodes), dtype=np.float64)

    def get_argmax_fx_with_proj(self, XT, Omega_X):
        """
        get argmax with projected gradient descent
        :param XT: all initial point for all time slot
        :param Omega_XT: the feasible set
        :return:
        """
        # TODO: figure out those hyperparameters
        lr, max_iter, bound = 0.001, 2000, 5
        T = len(XT)

        # compute gradient for each x
        X = np.copy(XT)  # we dont't want to modify the XT
        for i in range(max_iter):
            X_pre = np.copy(X)
            # print("iteration={}, ".format(i))
            for t in range(T):
                indicator_x = np.zeros_like(XT[t])
                omega_x = Omega_X[t]
                indicator_x[list(omega_x)] = 1.0  # used for projection
                xt = X[t]
                grad_xt = self.get_loss_grad(X, t)
                xt = self._update_maximizer(grad_xt, indicator_x, xt, bound, lr)
                # print("t={}, {}".format(t, sorted(np.nonzero(xt)[0])))
                X[t] = xt

            if 0 == i % 100:
                score, score1, score2, score3 = self.get_obj_value(X)
                print('obj value={:.5f}, global_ems_value={:.5f}, penalty={:.5f}'.format(score, score1, score2))
                diff_norm_x_array = np.linalg.norm(X - X_pre)
                # print(diff_norm_x_array)

            diff_norm_x = np.linalg.norm(X - X_pre) ** 2
            if diff_norm_x <= 1e-6:
                break
        return X

    def get_loss_grad(self, X, t):
        """
        maximize objective function (the sum of EMS + penalty), note the correctness of the format of gradient
        :param x:
        :return:
        """
        feature_vector = self.feature_matrix[t]

        sum_X = 0.
        for x in X:
            sum_X += np.sum(x)

        sum_ct_x = 0.
        if 0.0 == sum_X:
            print('gradient_x: input x vector values are all zeros !!!')
            grad = [0.] * self.num_nodes
        else:
            for i, x in enumerate(X):
                sum_ct_x += np.dot(self.feature_matrix[i], x)
            grad = feature_vector / np.sqrt(sum_X) - .5 * sum_ct_x / np.power(sum_X, 1.5)
            # if t == 3:
            #     ind = 64
            #     print("feature {}: {}, first term {}, second term {}".format(ind, feature_vector[ind], feature_vector[ind]/np.sqrt(sum_X), -0.5 * sum_ct_x / np.power(sum_X, 1.5)))
            #     print("trade off previous: {}".format(- (2*self.trade_off*(X[t][ind]- X[t-1][ind]))))
            #     print("trade off after: {}".format(2*self.trade_off*(X[t+1][ind]- X[t][ind])))

        # print("we are in loss grad, and t= {}".format(t))
        if t == 0:
            # pass
            grad += 2 * self.trade_off * (X[t + 1] - X[t])
            # print("gradient in argmax: {}".format(grad))
        elif t == self.T - 1:
            grad -= 2 * self.trade_off * (X[t] - X[t - 1])
        else:
            grad -= 2 * self.trade_off * (X[t] - X[t - 1])
            grad += 2 * self.trade_off * (X[t + 1] - X[t])

        if np.isnan(grad).any():
            raise ('something is wrong in gradient of x.')
        return grad

    def _update_maximizer(self, grad_x, indicator_x, x, bound, step_size):
        normalized_x = (x + step_size * grad_x) * indicator_x  # update and project
        sorted_indices = np.argsort(normalized_x)[::-1]

        normalized_x[normalized_x <= 0.0] = 0.0
        num_non_posi = len(np.where(normalized_x == 0.0))
        normalized_x[normalized_x > 1.0] = 1.0
        if num_non_posi == len(x):
            print("siga-1 is too large and all values in the gradient are non-positive")
            for i in range(bound):
                normalized_x[sorted_indices[i]] = 1.0
        return normalized_x


class SumEMS(object):
    def __init__(self, feature_matrix=None, trade_off=10.):
        """
        In baojian's Java code, inputs are base and count, and base is used when solving argmax problem
        :param feature_vector:
        :param lambda_:
        """
        self.feature_matrix = feature_matrix
        self.T = len(feature_matrix)
        self.num_nodes = len(feature_matrix[0])
        self.trade_off = trade_off

    def get_ems_score(self, x, feature_vector):
        """
        get EMS value for current x
        :param x:
        :return:
        """
        sum_x = np.sum(x)
        # sum_x = len(np.nonzero(x))
        if sum_x == 0.:
            return 0.
        else:
            ct_x = np.dot(feature_vector, x)
            func_val = ct_x / np.sqrt(sum_x)  # function val should not consider regularization
        return func_val

    def get_sum_ems(self, X):
        score = 0.
        for t in range(self.T):
            xt = X[t]
            feature_vector = self.feature_matrix[t]
            score += self.get_ems_score(xt, feature_vector)
        return score

    def get_global_ems(self, X):
        x = []
        fea = []
        for t in range(self.T):
            x += list(X[t])
            fea += list(self.feature_matrix[t])
        global_ems_score = self.get_ems_score(x, fea)
        return global_ems_score

    def get_obj_value(self, X):
        score, score1, score2 = 0., 0., 0.
        score3 = 0.
        for t in range(self.T):
            xt = X[t]
            feature_vector = self.feature_matrix[t]
            score1 += self.get_ems_score(xt, feature_vector)
        # print("sum of EMS", score1)
        for t in range(1, self.T):
            xt = X[t]
            xt_minus_1 = X[t - 1]
            diff = xt - xt_minus_1
            score2 -= self.trade_off * (np.linalg.norm(diff) ** 2)
            score3 -= (np.linalg.norm(diff) ** 2)
        score = score1 + score2
        return score, score3

    def get_init_point_random(self):
        """
        create a random binary vector as initial vector for each x
        :return:
        """
        XT = []
        for t in range(self.T):
            # xt_0 = np.random.rand(self.num_nodes)
            # xt_0 = np.array(xt_0 >= 0.5, dtype=np.float64)
            xt_0 = np.ones(self.num_nodes, dtype=np.float64)
            XT.append(xt_0)
        return np.array(XT)
        # return np.ones((self.T, self.num_nodes), dtype=np.float64)

    def get_argmax_fx_with_proj(self, XT, Omega_X):
        """
        get argmax with projected gradient descent
        :param XT: all initial point for all time slot
        :param Omega_XT: the feasible set
        :return:
        """
        # TODO: figure out those hyperparameters
        lr, max_iter, bound = 0.05, 1000, 5
        T = len(XT)

        # compute gradient for each x
        X = np.copy(XT)  # we dont't want to modify the XT
        for i in range(max_iter):
            X_pre = np.copy(X)
            for t in range(T):
                indicator_x = np.zeros_like(XT[t])
                omega_x = Omega_X[t]
                indicator_x[list(omega_x)] = 1.0  # used for projection
                xt = X[t]
                grad_xt = self.get_loss_grad(X, t)
                xt = self._update_maximizer(grad_xt, indicator_x, xt, bound, lr)
                X[t] = xt
            diff_norm_x = np.linalg.norm(X - X_pre)
            if diff_norm_x <= 1e-6:
                break
        return X

    def get_loss_grad(self, X, t):
        """
        maximize objective function, note the correctness of the format of gradient
        :param x:
        :return:
        """
        x = X[t]
        feature_vector = self.feature_matrix[t]
        sum_x = np.sum(x)
        if 0.0 == sum_x:
            print('gradient_x: input x vector values are all zeros !!!')
            grad = [0.] * self.num_nodes
        else:
            ct_x = np.dot(feature_vector, x)
            grad = feature_vector / np.sqrt(sum_x) - .5 * ct_x / np.power(sum_x, 1.5)

        if t == 0:
            # pass
            xt_plus_1 = X[t + 1]
            grad += 2 * self.trade_off * (xt_plus_1 - x)
        elif t == self.T - 1:
            xt_minus_1 = X[t - 1]
            grad -= 2 * self.trade_off * (x - xt_minus_1)
        else:
            xt_plus_1 = X[t + 1]
            xt_minus_1 = X[t - 1]
            grad -= 2 * self.trade_off * (x - xt_minus_1)
            grad += 2 * self.trade_off * (xt_plus_1 - x)

        if np.isnan(grad).any():
            raise ('something is wrong in gradient of x.')
        return grad

    def _update_maximizer(self, grad_x, indicator_x, x, bound, step_size):
        normalized_x = (x + step_size * grad_x) * indicator_x  # update and project
        sorted_indices = np.argsort(normalized_x)[::-1]

        normalized_x[normalized_x <= 0.0] = 0.0
        num_non_posi = len(np.where(normalized_x == 0.0))
        normalized_x[normalized_x > 1.0] = 1.0
        if num_non_posi == len(x):
            print("siga-1 is too large and all values in the gradient are non-positive")
            for i in range(bound):
                normalized_x[sorted_indices[i]] = 1.0
        return normalized_x


class FuncEMS(object):
    def __init__(self, feature_vector=None, lambda_=10.):
        """
        In baojian's Java code, inputs are base and count, and base is used when solving argmax problem
        :param feature_vector:
        :param lambda_:
        """
        self.feature_vector = feature_vector
        self.num_nodes = len(feature_vector)
        self.lambda_ = lambda_

    def get_fun_val(self, x):
        # TODO: need to check if the input x is binary vector
        """
        :param x: binary vector
        :return: a scalar
        """
        sum_x = np.sum(x)
        # ct_x = np.dot(self.feature_vector, x) ** 2. # (c^Tx)^2/1^Tx, squared c^T_x is just for convenience of proof
        ct_x = np.dot(self.feature_vector, x)  # (c^Tx)/\sqrt{1^Tx}
        # reg = 0.5 * self.lambda_ * (sum_x ** 2.)

        # func_val = ct_x / np.sqrt(sum_x) - reg
        func_val = ct_x / np.sqrt(sum_x)  # function val should not consider regularization ?

        return func_val

    def get_init_point_random(self):
        """
        create a random binary vector as initial vector
        :return:
        """
        # np.random.seed(seed=1)
        # x0 = np.random.rand(self.num_nodes)
        # x0 = np.array(x0 >= 0.5, dtype=np.float64)
        x0 = np.ones(self.num_nodes, dtype=np.float64)

        return x0

    def get_argmax_fx(self, omega_x):
        """
        get argmax with sorting heuristic
        :param omega_x:
        :return:
        """
        xt = np.zeros(self.num_nodes)
        # print('omega x', omega_x)
        # sort the omega_x based on the value of feature
        sorted_omega_x = [tup[0] for tup in sorted([(i, self.feature_vector[i]) for i in omega_x],
                                                   key=lambda tup: tup[1], reverse=True)]

        max_func_val = 0.
        # TODO: sort according to count/base, while not count only
        for k in range(1, len(omega_x) + 1):
            subset = sorted_omega_x[:k]
            # print(k, subset)
            x = np.zeros(self.num_nodes)
            x[subset] = 1.
            func_val = self.get_fun_val(x)
            if func_val > max_func_val:
                xt = x

        return xt

    def get_argmax_fx_with_proj(self, x0, omega_x):
        """
        get argmax with projected gradient descent
        :param x0: initial point
        :param omega_x: the feasible set
        :return:
        """
        # TODO: figure out those hyperparameters
        lr, max_iter, bound = 0.01, 1000, 5
        xt = np.copy(x0)
        # print("xt inside", xt)
        indicator_x = np.zeros_like(x0)
        indicator_x[list(omega_x)] = 1.0  # used for projection
        for i in range(max_iter):
            grad_x = self.get_loss_grad(xt)
            # print("inside grad",i, grad_x)
            # break
            x_pre = np.copy(xt)
            xt = self._update_maxmizer(grad_x, indicator_x, xt, bound, lr)
            diff_norm_x = np.linalg.norm(xt - x_pre)
            if diff_norm_x <= 1e-6:
                break
        # print("bx", xt)
        return xt

    def get_loss_grad(self, x):
        """
        maximize objective function, note the correctness of the format of gradient
        :param x:
        :return:
        """
        sum_x = np.sum(x)
        if 0.0 == sum_x:
            print('gradient_x: input x vector values are all zeros !!!')

        ct_x = np.dot(self.feature_vector, x)
        # reg = .5 * (self.lambda_ * (sum_x ** 2.))
        # grad = ct_x / np.sqrt(sum_x) - reg
        grad = self.feature_vector / np.sqrt(sum_x) - .5 * ct_x / np.power(sum_x, 1.5)

        if np.isnan(grad).any():
            print('something is wrong in gradient of x.')
        return grad

    def _update_maxmizer(self, grad_x, indicator_x, x, bound, step_size):
        normalized_x = (x + step_size * grad_x) * indicator_x  # update and project
        sorted_indices = np.argsort(normalized_x)[::-1]

        normalized_x[normalized_x <= 0.0] = 0.0
        num_non_posi = len(np.where(normalized_x == 0.0))
        normalized_x[normalized_x > 1.0] = 1.0
        if num_non_posi == len(x):
            print("siga-1 is too large and all values in the gradient are non-positive")
            for i in range(bound):
                normalized_x[sorted_indices[i]] = 1.0
        return normalized_x


def print_log(log_file, string):
    if logger == True:
        # print(string)
        if log_file != None:
            outfile = open(log_file, "a")
            outfile.write(string)
            outfile.close()
        else:
            print(string)


def normalized_gradient(x, grad):
    # rescale gradient to a feasible space [0, 1]?
    normalized_grad = np.zeros_like(grad)
    for i in range(len(grad)):
        if grad[i] > 0.0 and x[i] == 1.0:
            normalized_grad[i] = 0.0
        elif grad[i] < 0.0 and x[i] == 0.0:
            normalized_grad[i] = 0.0
        else:
            normalized_grad[i] = grad[i]
    return normalized_grad


def dynamic_graph_mp(data, k, max_iter, trade_off, log_file, func_name="EMS"):
    """
    :param func_name: score function name
    :param k: sparsity
    :param max_iter: max number of iterations
    :param G: networkx graph
    :param true_subgraph: a list of nodes that represents the ground truth subgraph
    :return: prediction xt, which denotes by the predicted abnormal nodes
    """
    if func_name == "SumEMS":
        features = data["features"]
        func = SumEMS(feature_matrix=features, trade_off=trade_off)
    elif func_name == "GlobalEMS":
        features = data["features"]
        func = GlobalEMS(feature_matrix=features, trade_off=trade_off)
    # elif func_name == "LaplacianEMS":
    #     features = data["features"]
    #     func = LaplacianEMS.LaplacianEMS(feature_matrix=features, trade_off=trade_off)
    else:
        print("ERROR")

    G = data["graph"]
    num_nodes = G.number_of_nodes()
    # costs = data["weights"]
    costs = np.ones(G.number_of_edges())
    T = len(data["subgraphs"])
    true_subgraphs = data["subgraphs"]
    edges = np.array(G.edges())

    print_log(log_file, "\n----------------initialization---------------\n")
    X = func.get_init_point_random()
    XT = np.copy(X)
    #
    print_log(log_file, "\n------------------searching------------------\n")
    for iter in range(max_iter):
        Omega_X = []
        X_prev = np.copy(XT)
        print("iter: {}, time: {}".format(iter, time.asctime(time.localtime(time.time()))))
        for t in range(T):
            xt = XT[t]
            grad = func.get_loss_grad(XT, t)

            if 0 == iter:
                xt_zero = np.zeros_like(xt)
                normalized_grad = normalized_gradient(xt_zero, grad)  # rescale gradient of x into [0, 1]
            else:
                normalized_grad = normalized_gradient(xt, grad)  # rescale gradient of x into [0, 1]

            # g: number of connected component
            re_head = head_proj(edges=edges, weights=costs, x=normalized_grad, g=1, s=k, budget=k - 1, delta=1. / 169.,
                                max_iter=100,
                                err_tol=1e-8, root=-1, pruning='strong', epsilon=1e-10, verbose=0)
            re_nodes, re_edges, p_x = re_head
            gamma_xt = set(re_nodes)
            supp_xt = set([ind for ind, _ in enumerate(xt) if _ != 0.])

            omega_x = gamma_xt.union(supp_xt)
            if 0 == iter:
                omega_x = gamma_xt
            Omega_X.append(omega_x)

        BX = func.get_argmax_fx_with_proj(XT, Omega_X)  # TODO: how to solve this argmax correctly

        for t in range(T):
            bx = BX[t]
            re_tail = tail_proj(edges=edges, weights=costs, x=bx, g=1, s=k, budget=k - 1, nu=2.5, max_iter=100,
                                err_tol=1e-8, root=-1,
                                pruning='strong', verbose=0)
            re_nodes, re_edges, p_x = re_tail
            psi_x = re_nodes
            xt = np.zeros_like(XT[t])
            xt[list(psi_x)] = bx[list(psi_x)]
            XT[t] = xt
        gap_x = np.linalg.norm(XT - X_prev) ** 2
        if gap_x < 1e-6:
            break

        print_log(log_file, '\ncurrent performance iteration: {}\n'.format(iter))
        if func_name == "SumEMS":
            obj_val, nonoverlap_penalty = func.get_obj_value(XT)
            ems_score = func.get_sum_ems(XT)
        elif func_name == "GlobalEMS":
            obj_val, ems_score, smooth_penalty, binarized_penalty = func.get_obj_value(XT)
        print_log(log_file, 'trade-off: {}\n'.format(trade_off))
        print_log(log_file, 'objective value of prediction: {:5f}\n'.format(obj_val))
        print_log(log_file, 'global ems score of prediction: {:5f}\n'.format(ems_score))
        print_log(log_file, 'penalty of prediction: {:5f}\n'.format(obj_val - ems_score))

        pred_subgraphs = [np.nonzero(x)[0] for x in XT]
        print_log(log_file, "----------------- current predicted subgraphs:\n")
        for t in range(T):
            pred_sub = sorted(pred_subgraphs[t])
            print_log(log_file, "{}, {}\n".format(t, pred_sub))

        print_log(log_file, "---------------------------------------------:\n")
        for t in range(T):
            pred_sub = sorted(pred_subgraphs[t])
            x = np.round(XT[t][pred_sub], 5)
            fea = np.round(features[t][pred_sub], 5)
            print_log(log_file, "{}, {}\n".format(t, zip(pred_sub, x, fea)))

        print_log(log_file, "----------------- current true subgraphs:\n")
        for t in range(T):
            true_sub = sorted(true_subgraphs[t])
            x = np.round(XT[t][true_sub], 5)
            fea = np.round(features[t][true_sub], 5)
            print_log(log_file, "{}, {}\n".format(t, zip(true_sub, x, fea)))

        global_prec, global_rec, global_fm, global_iou, valid_prec, valid_rec, valid_fm, valid_iou = evaluate(
            true_subgraphs, pred_subgraphs)
        print_log(log_file,
                  'global_prec={:4f},\nglobal_rec={:.4f},\nglobal_fm={:.4f},\nglobal_iou={:.4f}\n'.format(global_prec,
                                                                                                          global_rec,
                                                                                                          global_fm,
                                                                                                          global_iou))
        print_log(log_file,
                  'valid_prec={:.4f},\nvalid_rec={:.4f},\nvalid_fm={:.4f},\nvalid_iou={:.4f}\n'.format(valid_prec,
                                                                                                       valid_rec,
                                                                                                       valid_fm,
                                                                                                       valid_iou))

    return XT


def worker(para):
    (data, data_type, case_id, func_name, max_iter, trade_off, sparsity, log_file, result_file) = para

    G = data["graph"]
    features = data["features"]
    true_subgraphs = data["subgraphs"]
    T = len(true_subgraphs)
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    print_log(log_file, "---------------------------------Ground-Truth---------------------------------------\n")
    print_log(log_file, 'num of node: {}\n'.format(num_nodes))
    print_log(log_file, 'num of edges: {}\n'.format(num_edges))
    print_log(log_file, 'number of time stamps: {}\n'.format(T))
    print_log(log_file, 'all true subgraph: {}\n'.format(true_subgraphs))
    # print_log(log_file, 'all true subgraph sizes: {}\n'.format(data["true_subgraph_sizes"]))

    true_X = []
    for true_sub in true_subgraphs:
        true_x = np.zeros(num_nodes)
        true_x[true_sub] = 1.0
        true_X.append(true_x)

    if func_name == "SumEMS":
        func = SumEMS(feature_matrix=data["features"], trade_off=trade_off)
        obj_val, nonoverlap_penalty = func.get_obj_value(true_X)
        ems_score = func.get_sum_ems(true_X)
    elif func_name == "GlobalEMS":
        func = GlobalEMS(feature_matrix=data["features"], trade_off=trade_off)
        obj_val, ems_score, smooth_penalty, binarized_penalty = func.get_obj_value(true_X)
    print_log(log_file, '\ntrade-off: {}\n'.format(trade_off))
    print_log(log_file, 'objective value of ground-truth: {:5f}\n'.format(obj_val))
    print_log(log_file, 'global eml score of ground-truth: {:5f}\n'.format(ems_score))
    print_log(log_file, 'penalty of ground-truth: {:5f}\n'.format(obj_val - ems_score))

    print_log(log_file, "\n-----------------------------Dynamic Graph-MP--------------------------------------")
    XT = dynamic_graph_mp(data, sparsity, max_iter, trade_off, log_file, func_name)

    print_log(log_file, "\n--------------------------Evaluation of Raw Prediction-------------------------------")
    raw_pred_subgraphs = [np.nonzero(x)[0] for x in XT]
    global_prec, global_rec, global_fm, global_iou, valid_prec, valid_rec, valid_fm, valid_iou = evaluate(
        true_subgraphs, raw_pred_subgraphs, log_file)
    if func_name == "SumEMS":
        obj_val, nonoverlap_penalty = func.get_obj_value(XT)
        ems_score = func.get_sum_ems(XT)
    elif func_name == "GlobalEMS":
        obj_val, ems_score, smooth_penalty, binarized_penalty = func.get_obj_value(XT)
    elif func_name == "BlockEMS":
        obj_val, nonoverlap_penalty = func.get_obj_value(XT)
        ems_score = func.get_global_ems(XT)
    print_log(log_file, '\ntrade-off: {}\n'.format(trade_off))
    print_log(log_file, 'objective value of prediction: {:5f}\n'.format(obj_val))
    print_log(log_file, 'global ems score of prediction: {:5f}\n'.format(ems_score))
    print_log(log_file, 'smooth penalty of prediction: {:5f}\n'.format(obj_val - ems_score))

    print_log(log_file, "\n--------------------------Evaluation of Refined Prediction-------------------------------")
    edges, newid_oldid_dict = generate_combined_subgraphs(G, T, raw_pred_subgraphs)
    combined_G = nx.Graph()
    combined_G.add_edges_from(edges)
    largest_cc = max(nx.connected_components(combined_G), key=len)
    refined_pred_subgraphs = refine_predicted_subgraph(largest_cc, newid_oldid_dict, T)
    refined_XT = []
    for t in range(T):
        x = np.zeros_like(XT[t])
        pred_sub = sorted(refined_pred_subgraphs[t])
        x[pred_sub] = XT[t][pred_sub]
        refined_XT.append(x)
    global_prec, global_rec, global_fm, global_iou, valid_prec, valid_rec, valid_fm, valid_iou = evaluate(
        true_subgraphs, refined_pred_subgraphs, log_file)
    obj_val, ems_score, smooth_penalty, binarized_penalty = func.get_obj_value(refined_XT)
    print_log(log_file, '\nsmooth trade-off: {}\n'.format(trade_off))
    print_log(log_file, 'objective value of prediction: {:5f}\n'.format(obj_val))
    print_log(log_file, 'global ems score of prediction: {:5f}\n'.format(ems_score))
    print_log(log_file, 'temporal smooth penalty of prediction: {:5f}\n'.format(smooth_penalty))

    print('{}, {}, {}, {:.5f}, {:.5f}, {:.5f}\n'.format(
        trade_off, sparsity, case_id, global_prec, global_rec, global_fm))


def single_test():
    graph_type = "BA"
    num_nodes = 100
    func_name = "GlobalEMS"
    max_iter = 10
    mu1 = 5

    overlap = 0.5
    pattern = "m"
    data_type = "test"

    num_time_slots = 7
    windows = 5
    subsize_min, subsize_max = 100, 300
    start_time = int(num_time_slots / 2.0) - int(windows / 2.0)
    end_time = int(num_time_slots / 2.0) + int(windows / 2.0)
    case_id = 0

    num_instances = 1
    num_nodes = 100
    num_time_stamps = 9
    num_time_stamps_signal = 5
    start_time_stamps = num_time_stamps / 2 - num_time_stamps_signal / 2
    end_time_stamps = num_time_stamps / 2 + num_time_stamps_signal / 2
    num_nodes_subgraph_min = 10
    num_nodes_subgraph_max = 20
    overlap_ratio = 0.8
    mu_0 = 0.
    mu_1 = 5.

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/synthetic'
    fn = 'syn_mu_{:.1f}_{:.1f}_num_{:d}_{:d}_{:d}_{:d}_time_{:d}_{:d}_{:d}.pkl'.format(mu_0, mu_1, num_instances, num_nodes, num_nodes_subgraph_min, num_nodes_subgraph_max, start_time_stamps, end_time_stamps, num_time_stamps)
    data_dir = os.path.join(path, fn)
    print(data_dir)
    all_data = pickle.load(open(data_dir, "rb"))
    data = all_data[case_id]

    input_paras = []
    for trade_off in [0.01]:
        # for trade_off in [100]:
        for sparsity in [10]:
            log_file = None
            result_file = None
            para = (data, data_type, case_id, func_name, max_iter, trade_off, sparsity, log_file, result_file)
            input_paras.append(para)
            worker(para)
    # pool = Pool(processes=50)
    # pool.map(worker, input_paras)
    # pool.close()
    # pool.join()


if __name__ == "__main__":
    single_test()