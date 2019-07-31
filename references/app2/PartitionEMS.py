import numpy as np
import random
from multiprocessing import Pool, Process, Manager
import time
import copy_reg
import types
# import pathos.pools as pp

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)



class PartitionEMS(object):
    def __init__(self, features=None, num_blocks=None, nodes_set=None, boundary_edges_dict = None, nodes_id_dict =None, trade_off=10.):
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
        self.current_X = None
        self.current_Y = None
        self.current_Z = None
        self.Omega_X = None
        self.current_Z_next = None


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
        # sum_x = len(np.nonzero(x))
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
                score2 -= self.trade_off * (np.linalg.norm(diff)**2)
                score3 -= np.linalg.norm(diff)**2
        score = score1 + score2
        return score, score1, score2, score3


    def get_init_point_random(self):
        """
        create a random binary vector as initial vector for each x
        :return:
        """
        X = np.zeros(self.num_nodes, dtype=np.float64) + 0.000001
        return X


    def get_argmax_fx_with_proj(self, XT, Omega_X):
        """
        get argmax with projected gradient descent
        :param XT: all points for all blocks
        :param Omega_XT: the feasible set
        :return:
        """
        # TODO: figure out those hyperparameters
        lr, max_iter, bound = 0.001, 2000, 5

        # compute gradient for each x
        X = np.copy(XT)  # we dont't want to modify the XT
        for i in range(max_iter):
            X_pre = np.copy(X)
            for t in range(self.num_blocks):
                indicator_x = np.zeros_like(X[self.nodes_set[t]])
                omega_x = Omega_X[t]
                indicator_x[list(omega_x)] = 1.0  # used for projection
                xt = X[sorted(self.nodes_set[t])]
                fea = self.features[sorted(self.nodes_set[t])]
                boundary_xs_dict = self.get_boundary_xs(X, self.boundary_edges_dict[t], self.nodes_id_dict) #key is boundary edge, value is adjacent x in other blocks
                grad_xt = self.get_loss_grad(xt, fea, boundary_xs_dict, X)
                xt = self._update_maximizer(grad_xt, indicator_x, xt, bound, lr)
                # print("t={}, {}".format(t, sorted(np.nonzero(xt)[0])))
                X[sorted(self.nodes_set[t])] = xt
            diff_norm_x = np.linalg.norm(X - X_pre) ** 2
            if diff_norm_x <= 1e-6:
                break
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


    # def get_argmax_fx_with_proj_accelerated(self, XT, Omega_X):
    #     """
    #     get argmax with projected gradient descent
    #     :param XT: all points for all blocks
    #     :param Omega_XT: the feasible set
    #     :return:
    #     """
    #     # TODO: figure out those hyperparameters
    #     lr, max_iter, bound = 0.001, 2000, 5
    #     p = 1.0
    #
    #     # compute gradient for each x
    #     X = np.copy(XT)  # we dont't want to modify the XT
    #     for i in range(max_iter):
    #         X_pre = np.copy(X)
    #         for t in range(self.num_blocks):
    #             indicator_x = np.zeros_like(X[self.nodes_set[t]])
    #             omega_x = Omega_X[t]
    #             indicator_x[list(omega_x)] = 1.0  # used for projection
    #             xt = X[sorted(self.nodes_set[t])]
    #             fea = self.features[sorted(self.nodes_set[t])]
    #             boundary_xs_dict = self.get_boundary_xs(X, self.boundary_edges_dict[t], self.nodes_id_dict) #key is boundary edge, value is adjacent x in other blocks
    #             grad_xt = self.get_loss_grad(xt, fea, boundary_xs_dict, X)
    #             p = (1+ np.sqrt(1+4*p**2)) / 2.0
    #             w = (p-1.0)/p
    #             xt = self._update_maximizer_accelerated(grad_xt, indicator_x, xt, bound, lr, w)
    #             # print("t={}, {}".format(t, sorted(np.nonzero(xt)[0])))
    #             X[sorted(self.nodes_set[t])] = xt
    #         diff_norm_x = np.linalg.norm(X - X_pre) ** 2
    #         if diff_norm_x <= 1e-6:
    #             break
    #     return X
    #
    # def _update_maximizer_accelerated(self, grad_x, indicator_x, x, bound, step_size, w):
    #     # normalized_x = (x + step_size * grad_x) * indicator_x  # update and project
    #     normalized_x = (x + w * step_size * grad_x) * indicator_x
    #     sorted_indices = np.argsort(normalized_x)[::-1]
    #
    #     normalized_x[normalized_x <= 0.0] = 0.0
    #     num_non_posi = len(np.where(normalized_x == 0.0))
    #     normalized_x[normalized_x > 1.0] = 1.0
    #     if num_non_posi == len(x):
    #         print("siga-1 is too large and all values in the gradient are non-positive")
    #         for i in range(bound):
    #             normalized_x[sorted_indices[i]] = 1.0
    #     return normalized_x


    def get_argmax_fx_with_proj_parallel(self, XT, Omega_X):
        """
        get argmax with projected gradient descent
        :param XT: all points for all blocks
        :param Omega_XT: the feasible set
        :return:
        """
        # TODO: figure out those hyperparameters
        # lr, max_iter, bound = 0.001, 20, 5
        max_iter = 2000
        tao = 50 #number of cpus
        n = self.num_blocks
        theta = tao / float(n)

        # compute gradient for each x
        self.current_X = np.copy(XT)  # we dont't want to modify the XT
        self.current_Z = np.copy(self.current_X)
        self.Omega_X = Omega_X
        obj_val_list = []


        for k in range(max_iter):
            print("iteration {}, time: {}".format(k, time.asctime(time.localtime(time.time()))))
            X_pre = np.copy(self.current_X)
            self.current_Y = (1 - theta)*self.current_X + theta * self.current_Z
            # sample_block_size = random.sample(range(1, self.num_blocks+1), 1)[0]
            # S = random.sample(range(self.num_blocks), sample_block_size)
            S = random.sample(range(self.num_blocks), tao)
            # print("random sample blocks", S)
            self.current_Z_next = np.copy(self.current_Z)
            para_list = []
            block_outputs = []
            print("assignment start {}".format(time.time()))
            for block_index in S:
                # nodes = self.nodes_set[block_index]
                # fea = self.features[sorted(nodes)]
                # indicator_x = np.zeros_like(X[nodes])
                # omega_x = Omega_X[block_index]
                # indicator_x[list(omega_x)] = 1.0  # used for projection
                # z = Z_next[nodes]
                # x = X[nodes]
                # para = (block_index, nodes, fea, Y, z, x, theta, tao, n, indicator_x)
                para = (block_index, theta, tao, n, self.nodes_set, self.current_X, self.current_Y, self.current_Z, self.Omega_X)
                para_list.append(para)
                # block_index, output = self.block_worker(para)
                # block_outputs.append((block_index, output))

            pool = Pool(processes=50)
            # print("assignment end {}".format(time.time()))
            block_outputs = pool.map(self.block_worker, para_list)
            pool.close()
            pool.join()
            print("parallel end {}".format(time.time()))

            for (block_index, output) in block_outputs:
                self.current_Z_next[self.nodes_set[block_index]] = output
            self.current_X = self.current_Y + n / tao * theta * (self.current_Z_next - self.current_Z)
            theta = (np.sqrt(theta**4 + 4*theta**2) - theta**2 ) / 2.0
            self.current_Z = self.current_Z_next
            # print("k: {}, obj_val {}".format(k, self.get_obj_value(X, self.boundary_edges_dict)))
            diff_norm_x = np.linalg.norm(self.current_X - X_pre) ** 2
            if diff_norm_x <= 1e-6:
                break
            # if k > 100 and k % 10 == 0:
            #     obj_val = self.get_obj_value(self.current_X, self.boundary_edges_dict)
            #     if obj_val < np.mean(obj_val_list[-5:]):
            #         break
            #     else:
            #         obj_val_list.append(obj_val)
        return self.current_X


    def block_worker(self, para):
        # block_index, nodes, fea, Y, z, x, theta, tao, n, indicator_x = para
        block_index, theta, tao, n = para

        # print("block_index {}, start time: {}".format(block_index, time.asctime(time.localtime(time.time()))))
        nodes = self.nodes_set[block_index]
        fea = self.features[sorted(nodes)]
        indicator_x = np.zeros(len(nodes))
        omega_x = self.Omega_X[block_index]
        indicator_x[list(omega_x)] = 1.0  # used for projection
        z = self.current_Z_next[nodes]
        x = self.current_X[nodes]

        y = self.current_Y[nodes]
        v = 1000.0
        boundary_xs_dict = self.get_boundary_xs(self.current_Y, self.boundary_edges_dict[block_index], self.nodes_id_dict) #key is boundary edge, value is adjacent x in other blocks
        grad_y = self.get_loss_grad(y, fea, boundary_xs_dict)
        step_size = tao / (n * theta * v)
        z_next = self._update_maximizer(grad_y, indicator_x, z, bound=5, step_size=step_size)

        # print("block_index {}, step_size {}, end time: {}".format(block_index, step_size, time.asctime(time.localtime(time.time()))))
        return block_index, z_next





    #-----------------------------------------------------------------------------------------------
    def get_argmax_fx_with_proj_parallel_2(self, XT, Omega_X):
        """
        get argmax with projected gradient descent
        :param XT: all points for all blocks
        :param Omega_XT: the feasible set
        :return:
        """
        # TODO: figure out those hyperparameters
        # lr, max_iter, bound = 0.001, 20, 5
        max_iter = 2000
        tao = 20 #number of cpus
        n = self.num_blocks
        theta = tao / float(n)

        # compute gradient for each x
        X = np.copy(XT)  # we dont't want to modify the XT
        Z = np.copy(X)
        v = 1.0
        obj_val_list = []

        for k in range(max_iter):
            print("iteration {}, time: {}".format(k, time.asctime(time.localtime(time.time()))))
            X_pre = np.copy(X)
            Y = (1 - theta)*X + theta * Z
            # sample_block_size = random.sample(range(1, self.num_blocks+1), 1)[0]
            # S = random.sample(range(self.num_blocks), sample_block_size)
            S = random.sample(range(self.num_blocks), tao)
            # print("random sample blocks", S)
            Z_next = np.copy(Z)
            para_list = []
            # block_outputs = []
            manager = Manager()
            return_dict = manager.dict()
            jobs = []
            # print("assignment start {}".format(time.time()))
            for block_index in S:
                nodes = self.nodes_set[block_index]
                fea = self.features[sorted(nodes)]
                indicator_x = np.zeros_like(X[nodes])
                omega_x = Omega_X[block_index]
                indicator_x[list(omega_x)] = 1.0  # used for projection
                boundary_xs_dict = self.get_boundary_xs(Y, self.boundary_edges_dict[block_index], self.nodes_id_dict) #key is boundary edge, value is adjacent x in other blocks
                y = Y[nodes]
                z = Z_next[nodes]
                x = X[nodes]
                para = (block_index, nodes, fea, boundary_xs_dict, y, z, x, theta, tao, n, v, indicator_x)
                para_list.append(para)
                p = Process(target=self.block_worker_2, args=(para, return_dict))
                jobs.append(p)
                p.start()

            for proc in jobs:
                proc.join()
                # block_index, output = self.block_worker(para)
                # block_outputs.append((block_index, output))
            # pool = Pool(processes=tao)
            # block_outputs = pool.map(self.block_worker_2, para_list)
            # pool.close()
            # pool.join()
            # print("parallel end {}".format(time.time()))

            for block_index in return_dict.keys():
                # print("block index ", block_index)
                Z_next[self.nodes_set[block_index]] = return_dict[block_index]
            # for (block_index, output) in block_outputs:
            #     Z_next[self.nodes_set[block_index]] = output
            X = Y + n / tao * theta * (Z_next - Z)
            theta = (np.sqrt(theta**4 + 4*theta**2) - theta**2 ) / 2.0
            Z = Z_next

            diff_norm_x = np.linalg.norm(X - X_pre) ** 2
            if diff_norm_x <= 1e-6:
                print("k: {}, obj_val {}".format(k, self.get_obj_value(X, self.boundary_edges_dict)))
                break
            # obj_val = self.get_obj_value(X, self.boundary_edges_dict)
            # if obj_val < np.mean(obj_val_list[-10:]):
            #     break
            # else:
            #     obj_val_list.append(obj_val)
        return X


    def get_block_gradient(self, z, z_next, fea, y, boundary_xs_dict, theta, tao, n, v):
        grad = self.get_ems_grad(z_next, fea)
        grad -= self.get_penality_grad(y, boundary_xs_dict)
        grad -= float(n * theta * v)/ tao * (z_next - z)
        return grad


    def block_worker_2(self, para, return_dict):
        block_index, nodes, fea, boundary_xs_dict, y, z, x, theta, tao, n, v, indicator_x = para

        # print("block_index {}, start time: {}".format(block_index, time.asctime(time.localtime(time.time()))))
        z_next = np.copy(x)
        # z_next = np.zeros(len(x)) + 1e-6
        step_size = 0.001
        for k in range(2000):
            z_prev = np.copy(z_next)
            grad = self.get_block_gradient(z, z_next, fea, y, boundary_xs_dict, theta, tao, n, v)
            z_next = self._update_maximizer_block_2(grad, indicator_x, z_next, bound= 5, step_size=step_size)
            diff = np.linalg.norm(z_next - z_prev) **2
            if diff <= 1e-6:
                # print("block index {}, k {}".format(block_index, k))
                break
        # print("block_index {}, step_size {}, end time: {}".format(block_index, step_size, time.asctime(time.localtime(time.time()))))
        # return block_index, z_next
        return_dict[block_index] = z_next

    def _update_maximizer_block_2(self, grad_x, indicator_x, x, bound, step_size):
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



