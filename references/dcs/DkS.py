#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : DkS
# @Date     : 08/11/2019 14:41:41
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


import igraph as ig
import random
import numpy as np
import math
import numpy.linalg as LA
import span


def generateGraph(n, k):
    """
    low-rank approximation for DkS

    @param
    """
    g = ig.Graph.Full(k)
    g.add_vertices(n - k)

    for i in range(k, n):
        for j in range(2):
            target = np.random.randint(n)
            while target == i or g.vs[target].degree() == k - 2:
                target = np.random.randint(n)
            g.add_edge(i, target)
    g.simplify()

    print g.vcount()
    print g.ecount()

    return g


def negativeEigenvalMatrix(n, k, d):
    g = generateGraph(n, k)
    adj = g.get_adjacency()
    adj = list(adj)
    adj = np.array(adj)
    eig_vals, eig_vecs = LA.eig(adj)
    count = 0
    for val in eig_vals:
        if val > 0.0:
            count += 1
    while g.is_bipartite() or count >= d:
        g = generateGraph(n, k)
        adj = g.get_adjacency()
        adj = list(adj)
        adj = np.array(adj)
        eig_vals, eig_vecs = LA.eig(adj)
        count = 0
        for val in eig_vals:
            if val > 0.0:
                count += 1
    print adj
    return g


def LRADkS(graph, k, d):
    n = graph.vcount()
    Sd = []
    adj = graph.get_adjacency()
    # print adj
    adj = list(adj)
    # print adj.__class__.__name__
    adj = np.array(adj)

    eig_vals, eig_vecs = LA.eig(adj)
    count = 0
    for val in eig_vals:
        if val > 0.0:
            count += 1

    if graph.is_bipartite():
        print "This is Bipartite Graph"

    elif count >= d:
        print "Yes, Positive Eigenvalues"
        sigma = np.zeros((d, d))
        v, w, u = LA.svd(adj)

        for i in range(d):
            sigma[i][i] = w[i]
        # print "v", v
        # print "sigma",sigma
        # print "u", u
        """
        #test approx
        v = v[:,:d]
        u = u[:d,:]

        rec = np.around(v.dot(sigma).dot(u), decimals=3).tolist()
        print "low-rank approximation"
        for ele in rec:
            print ele
        """
        param1 = v[:, :d]
        param2 = w[:d]
        param2 = param2.T
        print param1.shape
        print param2.shape[0]

        maximum = float("-inf")
        xprime = None

        Sd, X = span.spannogram(param1, param2, eps=0.1, s=k)
        print len(Sd)
        print len(X)
        for i in range(len(Sd)):
            x = X[i]
            idx = Sd[i]
            print idx
            for l in idx[:-k]:
                # print l
                x[l] = 0
            # print x
            x /= np.linalg.norm(x)
            x = x.T

            value = x.dot(param1).dot(np.diag(param2)).dot(param1.T).dot(x.T)

            if value > maximum:
                xprime = x
                maximum = value

        print xprime
        result = []
        print "optimal Xd:", xprime
        for i in range(len(xprime[0])):
            if xprime[0][i] != 0.0:
                result.append(i)
        print "optimal subgraph:", result

    else:
        sigma = np.zeros((d, d))
        # print "Don't exist %d Positive Eigen-values", d
        delta = 0.5
        enumer = int(math.log(n, 2) / delta)
        print "enumer", enumer

        # competet the maximum
        maximum = float("-inf")
        for i in range(enumer):
            # graph_copy = graph.as_directed()

            adj_copy = np.array(adj)  # we need a copy of original matrix
            L = {}
            R = {}
            index1 = 0
            index2 = 0
            L_back = {}
            R_back = {}
            for j in range(n):
                edgelist = []
                coin = np.random.randint(2)
                if coin == 0:
                    L[j] = index1
                    L_back[index1] = j
                    index1 += 1
                else:
                    R[j] = index2
                    R_back[index2] = j
                    index2 += 1

            # print "L",L
            # print "L_back",L_back
            # print "R",R
            # print "R_back",R_back

            for node in L:
                neighbors = graph.neighbors(node)
                for nb in neighbors:
                    if nb in L:
                        # edgelist.append((node,nb))
                        adj_copy[node][nb] = 0.0
                        adj_copy[nb][node] = 0.0
            for node in R:
                neighbors = graph.neighbors(node)
                for nb in neighbors:
                    if nb in R:
                        # edgelist.append((node,nb))
                        adj_copy[node][nb] = 0.0
                        adj_copy[nb][node] = 0.0

                        # graph_copy.delete_edges(edgelist)
            # print graph_copy.ecount()
            # print adj_copy

            # extract bi-adjacency matrix
            B = []
            for l in range(n):
                if l in L:
                    b = []
                    for r in range(n):
                        if r in R:
                            b.append(adj_copy[l][r])
                    B.append(b)
            B = np.array(B)
            # print len(L)
            # print len(R)
            # print B.shape

            # if graph.is_bipartite():
            #    print "Yes, Bipartite Now"
            # else:
            #    print "NO, not a Bipartite Graph"

            v, w, u = LA.svd(B)
            for i in range(d):
                sigma[i][i] = w[i]

            param1 = v[:, :d]
            param2 = w[:d]
            param2 = param2.T
            Ud = u[:d, :]
            # print param1.shape
            # print param2.shape[0]

            for k1 in range(1, k):
                k2 = k - k1
                # print k2
                Sd, X = span.spannogram(param1, param2, eps=0.1, s=k1)
                for q in range(len(Sd)):
                    x = X[q]
                    idx = Sd[q]
                    # print idx

                    for l in idx[:-k1]:
                        x[l] = 0.0

                    x /= np.linalg.norm(x)
                    x = x.T
                    # print "x", x

                    val_i = x.dot(param1).dot(np.diag(param2)).dot(Ud).T
                    # print "x-shape",x.shape
                    # print "diag(param2)",np.diag(param2).shape
                    # print "param1",param1.shape
                    # print "val_i", val_i
                    idy = np.abs(val_i).argsort(axis=0)  # note abs here, not sure if needed
                    # idy = val_i.argsort(axis=0)
                    # print "idy",idy
                    for m in idy[:-k2]:
                        val_i[m] = 0
                    # print val_i.T
                    value = sum(val_i)[0]
                    # print value

                    if value > maximum:
                        result = []
                        # print "idx",(idx[-k1:].T)[0]
                        xindex = (idx[-k1:].T)[0]
                        xres = []
                        for xitem in xindex:
                            xres.append(L_back[xitem])
                        result += xres
                        # print result
                        yindex = (idy[-k2:].T)[0]
                        # print yindex
                        yres = []
                        for yitem in yindex:
                            yres.append(R_back[yitem])
                        # print "idy",(idy[-k2:].T)[0]
                        result += yres
                        print result
                        maximum = value

        # print result
        print "optimal subgraph:", result
        return result


if __name__ == "__main__":
    n = 100  # graph size
    k = 10  # densest k
    d = 2  # d eigenvalues

    # g = negativeEigenvalMatrix(n,k,d)

    # ig.plot(g)
    g = generateGraph(n, k)
    result = LRADkS(g, k, d)
    count = 0.0
    for res in result:
        if res < k:
            count += 1.0
    precision = count / k
    print precision