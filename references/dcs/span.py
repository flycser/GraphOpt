#!/usr/bin/env python
"""Contains the implementation of spannogram-based algorithms"""
import math
import numpy as np
from numpy import linalg

__author__ = 'Ioannis Mitliagkas'
__copyright__ = "Copyright 2014, Ioannis Mitliagkas"
__credits__ = ["Dimitris Papailiopoulos"]
__license__ = "MIT"
__version__ = "0.0.1"
__email__ = "ioannis@utexas.edu"
__status__ = "alpha"


def spannogram(u, w, eps=0.1, s=-1):
    """
    Runs the spannogram algorithm on a rank-d matrix.
    Uses the \epsilon-net argument
    :param u: A p x d array-like structure containing d orthonormal columns.
    :param w: A d x 1 array-like structure containing d eigenvalues.
    :param eps: Desired accuracy. Defaults to 0.1
    :param s: An integer describing the desired level of sparsity. A value of -1 signifies a dense component
    :rtype : object a p x 1 array-like structure, containing the optimal vector.
    """

    assert isinstance(u, np.ndarray)
    p = u.shape[0]
    d = u.shape[1]

    if s == -1:
        s = p

    assert w.shape[0] == d

    # maximum = float("-inf")
    # xprime = None

    Sd = []
    X = []
    for i in range(int(math.ceil(eps ** (-d)))):
        v = np.random.randn(d, 1)
        x = v.T.dot(np.sqrt(np.diag(w)).dot(u.T)).T

        """ 
        counter = 0
        for xi in x:
            if xi == 0.0:
                counter += 1
        while counter == p:
            v = np.random.randn(d, 1)
            x = v.T.dot( np.sqrt(np.diag(w)).dot(u.T) ).T
        """
        # print x.shape
        idx = np.abs(x).argsort(axis=0)
        # sort the x, and return the index list with coresponding to sorted list
        # print idx.shape
        # print idx[-s:]
        Sd.append(idx)
        X.append(x)
        """
        for l in idx[:-s]:
            #print l
            x[l] = 0


        #print x
        x /= np.linalg.norm(x)
        x=x.T

        value = x.dot(u).dot(np.diag(w)).dot(u.T).dot(x.T)
        if value > maximum:
            xprime = x
            maximum = value
    return xprime.T, maximum
    """
    return Sd, X
