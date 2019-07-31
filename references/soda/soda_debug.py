from __future__ import print_function
from networkx.classes.function import neighbors

__author__ = 'fengchen'

import itertools
from collections import defaultdict
import os
import time,math
import networkx as nx
import numpy as np
outlog=open("logging4.txt","a+")
def read_APDM_data(path):
    data = []
    nodes = []
    adjList = defaultdict(list)
    trueNodes = []
    trueFea = []
    lines = open(path).readlines()
    n = -1
    for idx, line in enumerate(lines):
        if line.strip().startswith('truth_features'):
            trueFea = map(int, str(line.strip().split('=')[1]).split())
            n = idx + 5
            break

    for idx in range(n, len(lines)):
        # print lines[idx]
        line = lines[idx]
        if line.find('END') >= 0:
            n = idx + 4
            break
        else:
            items = line.split(' ')
            nodes.append(int(items[0]))
            data.append(map(float, items[1:]))

    for idx in range(n, len(lines)):
        line = lines[idx]
        if line.find('END') >= 0:
            n = idx + 4
            break
        else:
            vertices = line.split(' ')
            edge = [0] * 2
            edge[0] = int(vertices[0])
            edge[1] = int(vertices[1])

            if edge[1] not in adjList[edge[0]]:
                adjList[edge[0]].append(edge[1])
            if edge[0] not in adjList[edge[1]]:
                adjList[edge[1]].append(edge[0])

    true_subgraph = []
    for idx in range(n, len(lines)):
        line = lines[idx]
        if line.find('END') >= 0:
            break
        else:
            items = line.split(' ')
            true_subgraph.append(int(items[0]))
            true_subgraph.append(int(items[1]))
    true_subgraph = sorted(list(set(true_subgraph)))
    return nodes, dict(adjList), data, true_subgraph, trueFea


"""
edges: {node_id: {neighbor_id}, ...}
W: n by p matrix
p: total number of attributes
C: tradeoff parameter
 
"""


def outlierScore(neighbors, W, p, edges, C, s):
    from cvxopt import matrix, solvers
    """ Returns outlier score"""
    pairs = list(itertools.combinations(neighbors, 2))
    Hm = []
    Lm = []
    w = []
    zeta = []
    b = []
    c = []

    for i in range(p):
        w.append([])
    for rg in range(len(pairs)):
        zeta.append([])

    j = 0
    for p in pairs:  # Equation 3.9-3.12
        n1 = min(p)
        n2 = max(p)

        if n2 in edges[n1]:  # Equation 3.9 and 3.10 (alternative lines for each)
            Hm.append(-1.0)
            Hm.append(0.0)
            Lm.append(0.0)
            Lm.append(0.0)
            for i in range(len(w)):
                w[i].append(1.0 * abs(W[n1][i] - W[n2][i]))
                w[i].append(0.0)
            for i in range(len(zeta)):
                if (i == j):
                    zeta[i].append(-1.0)
                    zeta[i].append(-1.0)
                else:
                    zeta[i].append(0.0)
                    zeta[i].append(0.0)

        else:  # Equation 3.11 and 3.12 (alternative lines for each)
            Hm.append(0.0)
            Hm.append(0.0)
            Lm.append(1.0)
            Lm.append(0.0)
            for i in range(len(w)):
                w[i].append(-1.0 * abs(W[n1][i] - W[n2][i]))
                w[i].append(0.0)
            for i in range(len(zeta)):
                if (i == j):
                    zeta[i].append(-1.0)
                    zeta[i].append(-1.0)
                else:
                    zeta[i].append(0.0)
                    zeta[i].append(0.0)

        b.append(0.0)
        b.append(0.0)

        j += 1
    for i in range(len(w)):  # Equation 3.13
        Hm.append(0.0)
        Hm.append(0.0)
        Lm.append(0.0)
        Lm.append(0.0)
        for i in range(len(zeta)):
            zeta[i].append(0.0)
            zeta[i].append(0.0)
        for j in range(len(w)):
            if (i == j):
                w[j].append(1.0)
                w[j].append(-1.0)
            else:
                w[j].append(0.0)
                w[j].append(0.0)
        b.append(1.0)
        b.append(0.0)

    # Equation 3.14
    Hm.append(0.0)
    Hm.append(0.0)
    Lm.append(0.0)
    Lm.append(0.0)
    for i in range(len(zeta)):
        zeta[i].append(0.0)
        zeta[i].append(0.0)
    for i in range(len(w)):
        w[i].append(1.0)
        w[i].append(-1.0)
    b.append(1.0)
    b.append(-1.0)

    # Now populating c, the order of coefficients in c is Hm, Lm, w, zeta
    c.append(1.0)
    c.append(-1.0)
    for i in w:
        c.append(0.0)
    for i in zeta:
        c.append(1.0 * C / len(pairs))

    tempList = []
    tempList.append(Hm)
    tempList.append(Lm)

    for i in w:
        tempList.append(i)
    for i in zeta:
        tempList.append(i)
    outlog.write(str(len(tempList))+" "+str(np.sum(tempList))+" non-zero index: "+" ".join(map(str,np.nonzero(tempList)[0])) +"\n")
    outlog.write(str(len(b))+" "+str(np.sum(b))+" non-zero index: "+" ".join(map(str,np.nonzero(b)[0])) +"\n")   
    outlog.write(str(len(c))+" "+str(np.sum(c))+" non-zero index: "+" ".join(map(str,np.nonzero(b)[0])) +"\n") 
    outlog.flush()    
    
    A = matrix(tempList)
    b = matrix(b)
    c = matrix(c)
    # Now feeding the input to simplex algo
    # print A
    # print b
    # print c
    solvers.options['maxiters'] = 100
    try:        
        sol = solvers.lp(c, A, b, solver='glpk',maxiters=100)
        # print sol['x']
    except Exception as e:
        print("error")
        print(e)
        time.sleep(1)
        return 0, None
    if sol['x'] == None:
        return None, None
    else:
        w1 = []
        for i in range(len(w)):
            w1.append([i, sol['x'][i + 2]])
        w1 = sorted(w1, key=lambda x: x[1])
        #print("w1:", w1)
        return sol['x'][0] - sol['x'][1], [w1[i][0] for i in range(s)]


def get_neighbors(edges, S):
    neighbors = set()
    for i in S:
        for j in edges[i]:
            neighbors.add(j)
    return neighbors


def load_crimes_of_chicago(file_path_):
    nodes = []
    adj_list = defaultdict(list)
    data = []
    true_subgraph = []
    true_features = []
    index = 1
    g = nx.Graph()
    with open(file_path_) as f:
        for each_line in f.readlines():
            if index == 1:
                n = int(each_line.rstrip().split(' ')[0])
                p = int(each_line.rstrip().split(' ')[1])
                for i in range(n):
                    nodes.append(i)
                    data.append([])
                index += 1
            elif 2 <= index <= (n + 1):
                for each_entry in each_line.rstrip().split(' '):
                    data[index - 2].append(float(each_entry))
                index += 1
            elif (index >= (n + 2)) and (index <= (n + 2)):
                num_edges = int(each_line.rstrip())
                index += 1
            elif (index >= (n + 3)) and (index <= (n + 3 + num_edges - 1)):
                endpoint0 = each_line.rstrip().split(' ')[0]
                endpoint1 = each_line.rstrip().split(' ')[1]
                edge = [0] * 2
                edge[0] = int(endpoint0)
                edge[1] = int(endpoint1)
                g.add_edge(edge[0], edge[1], weight=1.0)
                if edge[1] not in adj_list[edge[0]]:
                    adj_list[edge[0]].append(edge[1])
                if edge[0] not in adj_list[edge[1]]:
                    adj_list[edge[1]].append(edge[0])
                index += 1
            elif (index >= (n + num_edges + 3)) and (index <= (n + num_edges + 3)):
                for each_node in each_line.rstrip().split(' '):
                    true_subgraph.append(int(each_node))
                index += 1
            elif (index >= (n + num_edges + 4)) and (index <= (n + num_edges + 4)):
                for each_feature in each_line.rstrip().split(' '):
                    true_features.append(int(each_feature))
                index += 1
    print('number of nodes: {}'.format(n))
    print('number of features: {}'.format(p))
    print('true nodes size: {}'.format(len(true_subgraph)))
    print('true feature size: {}'.format(len(true_features)))
    return nodes, dict(adj_list), data, true_subgraph, true_features, g


def load_yelp_graph(file_path_):
    nodes = []
    adj_list = defaultdict(list)
    data = []
    true_subgraph = []
    true_features = []
    index = 0
    graph_mat_row_index = 0
    g = nx.Graph()
    with open(file_path_) as f:
        print(file_path_)
        for each_line in f.readlines():
            __items__ = each_line.rstrip().split(' ')
            if index == 0:
                n = int(__items__[0])
                p = int(__items__[1])
                k = int(__items__[2])
                s = int(__items__[3])
                num_edges = int(__items__[4])
                print('{} {} {} {} {}\n'.format(n, p, k, s, num_edges))
                nodes_dict = dict()
                words_dict = dict()
                for i in range(n):
                    nodes.append(i)
                    data.append([])
            elif 1 <= index <= n:
                for each_entry in __items__:
                    data[graph_mat_row_index].append(float(each_entry))
                graph_mat_row_index += 1
            elif (n + 1) <= index <= (2 * n):
                nodes_dict[__items__[0]] = int(__items__[1])
            elif (2 * n + 1) <= index <= (2 * n + p):
                words_dict[__items__[0]] = int(__items__[1])
            elif (2 * n + p + 1) == index:
                for each_node in __items__:
                    true_subgraph.append(int(each_node))
            elif index == (2 * n + p + 2):
                for each_feature in __items__:
                    true_features.append(int(each_feature))
            elif (2 * n + p + 3) <= index <= (2 * n + p + 3 + num_edges - 1):
                endpoint0 = each_line.rstrip().split(' ')[0]
                endpoint1 = each_line.rstrip().split(' ')[1]
                edge = [0] * 2
                edge[0] = int(endpoint0)
                edge[1] = int(endpoint1)
                g.add_edge(edge[0], edge[1], weight=1.0)
                if edge[1] not in adj_list[edge[0]]:
                    adj_list[edge[0]].append(edge[1])
                if edge[0] not in adj_list[edge[1]]:
                    adj_list[edge[1]].append(edge[0])
            index += 1
    print('number of nodes: {}'.format(n))
    print('number of features: {}'.format(p))
    print('true nodes size: {}'.format(len(true_subgraph)))
    print('true feature size: {}'.format(len(true_features)))
    words_reverse = dict()
    for item in words_dict:
        words_reverse[item] = words_dict[item]
    return nodes, dict(adj_list), data, true_subgraph, true_features, g, words_dict, words_reverse


"""
V: list of node ids that start from 0.
edges:
edges: {node_id: {neighbor_id}, ...}
W: n by p feature matrix [[] []]
radius: the radius each neighborhood
s: number of true attributes
C: a tradeoff parameter (100 by default)
"""


def get_candidates_nodes(true_nodes, g):
    import random
    initial_node = true_nodes[random.randint(0, len(true_nodes) - 1)]
    return true_nodes


def soda_scan(V, edges, W, radius, s, C, true_nodes, g):
    p = len(W[0])
    print(p)
    S_data = []

    for i in get_candidates_nodes(true_nodes, g):
        S = set()
        S.add(i)
        for step in range(radius):
            S = S.union(get_neighbors(edges, S))
        S_data.append(S)
    scores = []
    for i,S in enumerate(S_data[29:]):
        neighborhoods=get_neighbors(edges, S) 
        print('_____________________________ {}_________________________________'.format(i))
        if len(neighborhoods) < 100:  
            
            outlog.write(str(i)+" S_size="+str(len(S))+" neighborSize="+str(len(neighborhoods))+" S=["+" ".join(map(neighborhoods,S)) +"]\n")
            #outlog.write(str(i)+" S_size="+str(len(S))+" neighborSize="+str(len(neighborhoods)))#+" neighborhoods=["+" ".join(map(str,neighborhoods)) +"]\n")
            outlog.flush()
            
            for idx in sorted(neighborhoods):                
                outlog.write(str(idx)+" "+" ".join(map(str,W[idx])) +"\n")
                outlog.flush()          
            t0=time.time()
           
            [fvalue, feature_set] = outlierScore(neighborhoods, W, p, edges, C, s)
            outlog.write(str(round(time.time()-t0,2))+"\n")
            outlog.flush()
            if fvalue != None:
                scores.append([fvalue, S, feature_set])
                print(fvalue, S)
                #    sorted(scores, key=lambda x:x[0], reverse=True)
    #print(scores)
    if len(scores)==0:
        return 0.0,[],[]
    [fvalue, S, feature_set] = max(scores, key=lambda x: x[0])
    return fvalue, S, feature_set


def stat_calc(true_nodes, true_features, S, R):
    true_features = set(true_features)
    true_nodes = set(true_nodes)
    node_prec = len(true_nodes.intersection(S)) / (len(true_nodes) * 1.0)
    node_recall = len(true_nodes.intersection(S)) / (len(S) * 1.0)
    feature_prec = len(true_features.intersection(R)) / (len(true_features) * 1.0)
    feature_recall = len(true_features.intersection(R)) / (len(R) * 1.0)
    return node_prec, node_recall, feature_prec, feature_recall


def get_data_file_list(output_path_):
    file_list = []
    keywords = []
    for file_path in sorted(os.listdir(output_path_)):
        file_list.append(output_path_ + file_path)
        keywords.append(file_path.split('_')[1])
    return file_list, keywords


def unittest4():
    V, edges, W, true_nodes, true_features = read_APDM_data('data/test1.txt')
    print(V)
    # print dict(edges)
    print(W)
    print(true_nodes)
    print(true_features)
    s = len(true_features)
    C = 10
    [fvalue, S, R] = soda_scan(V, edges, W, 1, s, C)
    # print S, R
    # [node_prec, node_recall, feature_prec, feature_recall] = stat_calc(true_nodes, true_features, S, R)
    # print node_prec, node_recall, feature_prec, feature_recall


def test_on_yelp_data():
    file_path = 'F:/workspace/git/SG-Pursuit/data/Yelp'
    file_list, keywords = get_data_file_list(file_path)
    index = 0
    import time
    for each_file in file_list[19:]: 
        if "26" not in each_file:
            continue
        outlog.write(each_file+"\n")
        outlog.flush()       
        start_time = time.time()
        fp = open('./final_result_SODA_yelp_different_para_New_May9.txt', 'a')
        V, edges, W, true_nodes, true_features, g, words_dict, words_reverse = load_yelp_graph(each_file)
        outlog.write("True Nodes: "+ " ".joint(map(str,true_nodes))+"\n")
        outlog.flush() 
        type_ = keywords[index]
        result = []
        best_fm = -1.0
        for ridus in [2, 3, 4, 5]:
            outlog.write(str(ridus)+"\n")
            outlog.flush() 
            print('----------------------------{} {}---------------------------------'.format(each_file.split("_")[-1],ridus))
            C = 100.0
            s = len(true_features)
            [fvalue, S, R] = soda_scan(V, edges, W, ridus, s, C, true_nodes, g)
            if len(S)<1:
                continue
            print(S, R)
            [node_prec, node_recall, feature_prec, feature_recall] = stat_calc(true_nodes, true_features, S, R)
            print('-----------------------------------------------------')
            print(node_prec, node_recall, feature_prec, feature_recall)
            print('-----------------------------------------------------')
            if (node_prec + node_recall) == 0.0:
                node_fm = 0.0
            else:
                node_fm = 2.0 * (node_prec * node_recall) / (node_prec + node_recall)
            if (feature_prec + feature_recall) == 0.0:
                feature_fm = 0.0
            else:
                feature_fm = 2.0 * (feature_prec * feature_recall) / (feature_prec + feature_recall)
            if not result:
                result = [type_, str(node_prec), str(node_recall), str(node_fm)]
                result = result + [str(feature_prec), str(feature_recall), str(feature_fm)]
                best_fm = node_fm
            elif node_fm > best_fm:
                result = [type_, str(node_prec), str(node_recall), str(node_fm)]
                result = result + [str(feature_prec), str(feature_recall), str(feature_fm)]
                best_fm = node_fm
        run_time = time.time() - start_time
        run_time *= (len(V) * 1.0 / len(true_nodes))
        result.append('0 0 0.0 ' + str(run_time))
        nodes_str = ' '.join([words_reverse[fea_] for fea_ in R])
        index += 1
        fp.write(' '.join(result) + nodes_str + '\n')
        fp.close()
        print('finish ...')
        time.sleep(2)


if __name__ == '__main__':
    test_on_yelp_data()