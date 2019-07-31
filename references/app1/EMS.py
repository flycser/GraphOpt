import numpy as np



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
            diff = X[t] - X[t-1]
            score2 -= self.trade_off * (np.linalg.norm(diff)**2)
            score3 -= np.linalg.norm(diff)**2
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
            grad += 2 * self.trade_off * (X[t+1] - X[t])
            # print("gradient in argmax: {}".format(grad))
        elif t == self.T - 1:
            grad -= 2 * self.trade_off * (X[t] - X[t-1])
        else:
            grad -= 2 * self.trade_off * (X[t] - X[t-1])
            grad += 2 * self.trade_off * (X[t+1] - X[t])

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
            score2 -= self.trade_off * (np.linalg.norm(diff)**2)
            score3 -= (np.linalg.norm(diff)**2)
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
        #TODO: need to check if the input x is binary vector
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
        #TODO: figure out those hyperparameters
        lr, max_iter, bound = 0.01, 1000, 5
        xt = np.copy(x0)
        # print("xt inside", xt)
        indicator_x = np.zeros_like(x0)
        indicator_x[list(omega_x)] = 1.0 #used for projection
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
        normalized_x = (x + step_size * grad_x) * indicator_x  #update and project
        sorted_indices = np.argsort(normalized_x)[::-1]

        normalized_x[normalized_x <= 0.0] = 0.0
        num_non_posi = len(np.where(normalized_x == 0.0))
        normalized_x[normalized_x > 1.0] = 1.0
        if num_non_posi == len(x):
            print("siga-1 is too large and all values in the gradient are non-positive")
            for i in range(bound):
                normalized_x[sorted_indices[i]] = 1.0
        return normalized_x
