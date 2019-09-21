#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : densest_charikar_3
# @Date     : 08/11/2019 09:48:40
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :

from __future__ import print_function

import networkx as nx

from evaluation_metrics import degree_density, density, triangle_density


def greedy_degree_density(G):

    degrees = dict(G.degree())
    # print(degrees)
    sum_degrees = sum([degrees[node] for node in degrees])

    num_nodes = G.number_of_nodes()
    import operator
    # nodes = sorted(degrees, key=lambda tup: tup[1])
    nodes = sorted(degrees.items(), key=operator.itemgetter(1))
    print(nodes)
    nodes = [node[0] for node in nodes]
    # print(nodes)

    bin_boundaries = [0] # save start node of each degree
    curr_degree = 0
    for i, v in enumerate(nodes):
        if degrees[v] > curr_degree:
            bin_boundaries.extend([i] * (degrees[v] - curr_degree))
            print(i, v)
            print(bin_boundaries)
            print(len(bin_boundaries))
            curr_degree = degrees[v]

    print(bin_boundaries)
    node_pos = dict((v, pos) for pos, v in enumerate(nodes))
    print(node_pos)

    # neighbors = G.neighbors()

    nbrs = dict((v, set([node for node in G.neighbors(v)])) for v in G)

    max_degree_density = sum_degrees / float(num_nodes)
    ind = 0

    for v in nodes: # ascendingly degree
        num_nodes -= 1
        while degrees[v] > 0:
            pos = node_pos[v]
            bin_start = bin_boundaries[degrees[v]]
            node_pos[v] = bin_start
            node_pos[nodes[bin_start]] = pos
            nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
            bin_boundaries[degrees[v]] += 1
            degrees[v] -= 1

        for u in nbrs[v]:
            nbrs[u].remove(v)
            pos = node_pos[u]
            bin_start = bin_boundaries[degrees[u]]
            node_pos[u] = bin_start
            node_pos[nodes[bin_start]] = pos
            nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
            bin_boundaries[degrees[u]] += 1
            degrees[u] -= 1
            sum_degrees -= 2

        if num_nodes > 0:
            current_degree_density = sum_degrees / float(num_nodes)
            if current_degree_density > max_degree_density:
                max_degree_density = current_degree_density
                ind = G.number_of_nodes() - num_nodes

    optimal_nodes = nodes[ind:]

    return  G.subgraph(optimal_nodes)


def find_triangles(G):
    """
    For each node returns the number of triangles the node is part of
    and the pair of other nodes that form each of these triangles
    """
    triangles = {}
    nbrs = {}
    for node in G.nodes():
        triangles[node] = 0
        nbrs[node] = set()
        neighbors = G.neighbors(node)
        y = [node for node in neighbors]
        # print('y', neighbors[0])
        x = list(neighbors)
        # print('fei', x)
        # print(node)
        # print(G[node])
        # for node in neighbors:
        #     print('xxx', node)
        #     break
        # print('xxx')
        # for i in range(len(neighbors)):
        for i in range(len(x)):
            # for j in range(i + 1, len(neighbors)):
            for j in range(i + 1, len(x)):
                # if G.has_edge(neighbors[i], neighbors[j]):
                if G.has_edge(y[i], y[j]):
                    triangles[node] += 1
                    # if neighbors[i] < neighbors[j]:
                    if y[i] < y[j]:
                        # nbrs[node].add((neighbors[i], neighbors[j]))
                        nbrs[node].add((y[i], y[j]))
                    else:
                        # nbrs[node].add((neighbors[j], neighbors[i]))
                        nbrs[node].add((y[j], y[i]))

    return triangles, nbrs


def greedy_triangle_density(G):
    """
    Returns the subgraph with optimal triangle density
    """
    triangles, nbrs = find_triangles(G)
    sum_triangles = sum(triangles.values())
    num_nodes = G.number_of_nodes()
    nodes = sorted(triangles, key=triangles.get)
    bin_boundaries = [0]
    curr_triangle_number = 0
    for i, v in enumerate(nodes):
        if triangles[v] > curr_triangle_number:
            bin_boundaries.extend([i] * (triangles[v] - curr_triangle_number))
            curr_triangle_number = triangles[v]
    node_pos = dict((v, pos) for pos, v in enumerate(nodes))

    max_triangle_density = float(sum_triangles) / num_nodes
    ind = 0

    for v in nodes:

        num_nodes -= 1
        while triangles[v] > 0:
            pos = node_pos[v]
            bin_start = bin_boundaries[triangles[v]]
            node_pos[v] = bin_start
            node_pos[nodes[bin_start]] = pos
            nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
            bin_boundaries[triangles[v]] += 1
            triangles[v] -= 1
            sum_triangles -= 1

        for pair in nbrs[v]:

            if v < pair[1]:
                nbrs[pair[0]].remove((v, pair[1]))
            else:
                nbrs[pair[0]].remove((pair[1], v))

            pos = node_pos[pair[0]]
            bin_start = bin_boundaries[triangles[pair[0]]]
            node_pos[pair[0]] = bin_start
            node_pos[nodes[bin_start]] = pos
            nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
            bin_boundaries[triangles[pair[0]]] += 1
            triangles[pair[0]] -= 1
            sum_triangles -= 1

            if v < pair[0]:
                nbrs[pair[1]].remove((v, pair[0]))
            else:
                nbrs[pair[1]].remove((pair[0], v))

            pos = node_pos[pair[1]]
            bin_start = bin_boundaries[triangles[pair[1]]]
            node_pos[pair[1]] = bin_start
            node_pos[nodes[bin_start]] = pos
            nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
            bin_boundaries[triangles[pair[1]]] += 1
            triangles[pair[1]] -= 1
            sum_triangles -= 1

        if num_nodes > 0:
            current_triangle_density = float(sum_triangles) / num_nodes
            if current_triangle_density > max_triangle_density:
                max_triangle_density = current_triangle_density
                ind = G.number_of_nodes() - num_nodes

    optimal_nodes = nodes[ind:]

    return G.subgraph(optimal_nodes)


if __name__ == '__main__':

    fn = 'edges_4.txt'
    G = nx.read_edgelist(fn, delimiter=' ', nodetype=int)

    G = G.to_undirected()
    for node in G.nodes_with_selfloops():
        G.remove_edge(node, node)

    G1 = nx.Graph()
    for edge in G.edges():
        u = edge[0]
        v = edge[1]
        if u == v:
            continue
        if not G1.has_edge(u, v):
            G1.add_edge(u, v, weight=1.0)

    G = G1
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())
    print()

    subg = greedy_degree_density(G)
    print("----Greedy Degree Density----")
    print("Degree Density: " + str(degree_density(subg)))
    print("Density: " + str(density(subg)))
    print("Triangle Density: " + str(triangle_density(subg)))
    print("# Nodes: " + str(subg.number_of_nodes()))

    print(nx.density(G))
    print(nx.density(subg))
    print()


    subg = greedy_triangle_density(G)
    print("----Greedy Triangle Density----")
    print("Degree Density: " + str(degree_density(subg)))
    print("Density: " + str(density(subg)))
    print("Triangle Density: " + str(triangle_density(subg)))
    print("# Nodes: " + str(subg.number_of_nodes()))
    print()