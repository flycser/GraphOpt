import numpy as np
import random
from multiprocessing import Pool
import time
import copy_reg
import types


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

    def get_obj_value(self, X, boundary_edges_dict):
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
            grad = [0.] * self.num_nodes
        else:
            ct_x = np.dot(fea, xt)
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

    def get_loss_grad(self, xt, fea, boundary_xs_dict):
        grad = self.get_ems_grad(xt, fea)
        grad -= self.get_penality_grad(xt, boundary_xs_dict)

        if np.isnan(grad).any():
            raise ('something is wrong in gradient of x.')
        return grad


    def get_argmax_fx_with_proj_accelerated(self, XT, Omega_X):
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
                grad_xt = self.get_loss_grad(xt, fea, boundary_xs_dict)
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
                xt_hat = xt + w * (xt - xt_prev)
                X_hat[sorted(self.nodes_set[t])] = xt_hat
                X_prev[sorted(self.nodes_set[t])] = xt
                fea = self.features[sorted(self.nodes_set[t])]
                boundary_xs_dict = self.get_boundary_xs(X_hat, self.boundary_edges_dict[t], self.nodes_id_dict)  # key is boundary edge, value is adjacent x in other blocks
                xt = self.grad_ascent(xt_prev, xt_hat, fea, boundary_xs_dict, indicator_x)

                # fea = self.features[sorted(self.nodes_set[t])]
                # boundary_xs_dict = self.get_boundary_xs(X, self.boundary_edges_dict[t], self.nodes_id_dict)  # key is boundary edge, value is adjacent x in other blocks
                # xt = self.grad_ascent(xt, xt, fea, boundary_xs_dict, indicator_x)
                X[sorted(self.nodes_set[t])] = xt


            obj_val = self.get_obj_value(X, self.boundary_edges_dict)
            if i >= 1 and obj_val < obj_val_list[-1]:
                print("i: {}, obj_val {}".format(i, self.get_obj_value(X, self.boundary_edges_dict)))
                break
            obj_val_list.append(obj_val)
            # print("i: {}, obj_val {}".format(i, self.get_obj_value(X, self.boundary_edges_dict)))
            diff_norm_x = np.linalg.norm(X - X_prev) ** 2
            if i >= 1 and diff_norm_x <= 1e-6:
                print("i: {}, obj_val {}".format(i, self.get_obj_value(X, self.boundary_edges_dict)))
                break
        return X

    def get_block_grad(self, xt_hat, x_next, fea, boundary_xs_dict, step_size):
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
