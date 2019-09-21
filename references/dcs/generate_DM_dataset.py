#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : generate_DM_dataset
# @Date     : 07/31/2019 21:57:07
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :

from __future__ import print_function

import os
import time
import pickle
import string
import operator
import itertools

import nltk
import numpy as np
import networkx as nx

from nltk.corpus import stopwords
from scipy.stats.stats import pearsonr


def extract_info_file():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/DBLP/DBLP_Citation_2014_May/DM'
    fn = 'publications.txt'

    dm_confs = set(('KDD', 'ICDM', 'SDM', 'PKDD', 'CIKM'))

    count = 0
    authors = set()
    x = []
    with open(os.path.join(path, fn)) as rfile, open(os.path.join(path, 'dm.txt'), 'w') as wfile:
        while True:
            record = []
            line = rfile.readline()
            tag = False
            if line.startswith('#*'):
                title = line
                record.append(line)
                # wfile.write(line)
                while True:
                    line = rfile.readline()
                    record.append(line)

                    if line.startswith('#@'):
                        paper_authors = line.strip()[2:].split(',')
                        x.append(len(paper_authors))

                    if line.startswith('#t'):
                        year = line.strip()[2:]

                    if line.startswith('#c'):
                        conf = line.strip()[2:]
                        if conf in dm_confs and int(year) <= 2008:
                            tag = True
                            for author in paper_authors:
                                authors.add(author)
                                print(author)


                    if line.startswith('#!'):
                        break

            if tag:
                count += 1
                print(count)
                for line in record:
                    wfile.write(line)

            if not line:
                break

    print(count)

    print(len(authors))

    print(np.mean(x))

    fn = 'dm_authors.pkl'
    author_dict = {}
    with open(os.path.join(path, fn), 'wb') as wfile:
        id = 0
        for author in authors:
            author_dict[author] = id
            id += 1

        pickle.dump(author_dict, wfile)

def construct_coauthor_network():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/DBLP/DBLP_Citation_2014_May/DM'
    fn = 'dm_authors.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        dm_authors = pickle.load(rfile)

    graph = nx.Graph()

    fn = 'dm.txt'

    with open(os.path.join(path, fn)) as rfile:
        while True:
            record = []
            line = rfile.readline()
            if line.startswith('#*'):
                title = line
                record.append(line)
                while True:
                    line = rfile.readline()
                    record.append(line)

                    if line.startswith('#@'):
                        paper_authors = line.strip()[2:].split(',')
                        paper_author_ids = [dm_authors[author] for author in paper_authors]
                        x = list(itertools.combinations(paper_author_ids, 2))
                        for u, v in x:
                            graph.add_edge(u, v)

                    if line.startswith('#t'):
                        year = line.strip()[2:]

                    if line.startswith('#c'):
                        conf = line.strip()[2:]

                    if line.startswith('#!'):
                        break

            if not line:
                break

    print(graph.number_of_nodes())
    print(graph.number_of_edges())
    lcc = max(nx.connected_component_subgraphs(graph), key=len)
    subgraph = nx.subgraph(graph, lcc)
    print(subgraph.number_of_nodes())
    print(subgraph.number_of_edges())

    fn = 'dm_first_graph.pkl'
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(subgraph, wfile)

def word_list():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/DBLP/DBLP_Citation_2014_May/DM'
    fn = 'dm.txt'

    terms = list()
    count = 0
    with open(os.path.join(path, fn)) as rfile:
        while True:
            line = rfile.readline()
            if line.startswith('#*'):
                count += 1
                print(count)
                # if count = = 1023:
                #     print(line)
                line = line.strip()
                title = line[2:-1] if line.endswith('.') else line[2:]
                x = nltk.word_tokenize(title.lower())
                print(x)
                x = [term for term in x if term not in stopwords.words('english') and term not in string.punctuation]
                print(x)

                terms.extend(x)

            if not line:
                break

    print(len(terms))
    print(len(set(terms)))

    terms = list(set(terms))
    term_dict = {}
    for i, term in enumerate(terms):
        term_dict[term] = i

    # key word, valud word id
    fn = 'dm_words.pkl'
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(term_dict, wfile)

def construct_author_term_matrix():

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/DBLP/DBLP_Citation_2014_May/DM'
    fn = 'dm_words.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        words_dict = pickle.load(rfile)

    fn = 'dm_authors.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        dm_authors = pickle.load(rfile)


    fn = 'first_dm_graph.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        first_dm_graph = pickle.load(rfile)

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/DBLP/DBLP_Citation_2014_May'
    fn = 'dm.txt'


    author_term_mat = {}

    count = 0
    with open(os.path.join(path, fn)) as rfile:
        while True:
            line = rfile.readline()
            if line.startswith('#*'):
                count += 1
                # print(count)
                # if count == 1023:
                #     print(line)
                line = line.strip()
                title = line[2:-1] if line.endswith('.') else line[2:]
                x = nltk.word_tokenize(title.lower())
                x = [term for term in x if term not in stopwords.words('english') and term not in string.punctuation]

            if line.startswith('#@'):
                paper_authors = line.strip()[2:].split(',')
                paper_author_ids = [dm_authors[author] for author in paper_authors]

                for term in x:
                    for author in paper_author_ids:
                        if author not in author_term_mat and author in first_dm_graph.nodes():
                            author_term_mat[author] = np.zeros(len(words_dict))
                            author_term_mat[author][words_dict[term]] += 1
                        elif author in author_term_mat and author in first_dm_graph: # note !!!
                            author_term_mat[author][words_dict[term]] += 1

            if not line:
                break


    fn = 'dm_author_term_dict.pkl'
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(author_term_mat, wfile)


def test():

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/DBLP/DBLP_Citation_2014_May/DM'

    # first graph, original id
    fn = 'dm_first_graph.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        first_dm_graph = pickle.load(rfile)

    print(first_dm_graph.number_of_nodes())

    #
    fn = 'dm_author_term_dict.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        author_term_mat = pickle.load(rfile)

    print(len(author_term_mat))

    # key word, value word id
    fn = 'dm_words.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        words_dict = pickle.load(rfile)

    print(words_dict)


    print(author_term_mat.keys())

    print(np.nonzero(author_term_mat[2184]))

    fn = 'dm_authors.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        dm_authors = pickle.load(rfile)

    print(dm_authors['Xindong Wu'])

    for name, id  in dm_authors.items():
        if id == 2184:
            print(name)

    for word, id  in words_dict.items():
        if id in [ 39,   79,  185,  385,  420,  476,  492,  496,  640,  756,  943,
       1054, 1072, 1261, 1402, 1518, 1519, 1792, 1891, 1929, 1985, 2035,
       2067, 2167, 2256, 2450, 2535, 2548, 2619, 2647, 2648, 2655, 2751,
       2833, 2895, 3047, 3205, 3212, 3285, 3303, 3316, 3336, 3350, 3396,
       3411, 3483, 3516, 3540, 3610, 3674, 3697, 3930, 3981, 4025, 4110,
       4116, 4138, 4153, 4232, 4367, 4383, 4470, 4473, 4498, 4582, 4795,
       4851, 5053, 5088, 5179, 5233, 5360, 5404, 5439, 5471]:
            print(word, id, author_term_mat[2184][id])

def convert_id_from_zero():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/DBLP/DBLP_Citation_2014_May/DM'

    # first graph, original id
    fn = 'dm_first_graph.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        first_dm_graph = pickle.load(rfile)

    print(first_dm_graph.number_of_nodes())

    author_old_new_map = {}
    author_new_old_map = {}
    for id, node in enumerate(first_dm_graph.nodes()):
        author_old_new_map[node] = id
        author_new_old_map[id] = node

    fn = 'dm_author_map.pkl' # old id, new id
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump({'old_new': author_old_new_map, 'new_old': author_new_old_map}, wfile)

    fn = 'dm_author_term_dict.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        author_term_dict = pickle.load(rfile)

    print(len(author_new_old_map))
    print(author_new_old_map)

    string_map = {}
    for node in first_dm_graph.nodes():
        string_map[node] = '{}'.format(node)

    tmp_graph = nx.relabel_nodes(first_dm_graph, string_map)
    # print(string_map)

    string_old_new_map = {}
    for node in tmp_graph.nodes(): # tmp_graph, old id in string format
        # print(node)
        string_old_new_map[node] = author_old_new_map[int(node)]
        # print(node, author_old_new_map[int(node)])

    relabeled_dm_first_graph = nx.relabel_nodes(tmp_graph, string_old_new_map)
    # print(relabeled_dm_first_graph.number_of_nodes())
    author_term_mat = np.zeros((relabeled_dm_first_graph.number_of_nodes(), len(author_term_dict[2])))

    print(len(author_term_dict[2]))
    print(author_term_mat.shape)
    for node in relabeled_dm_first_graph.nodes(): # new id
        author_term_mat[node] = author_term_dict[author_new_old_map[node]]
        if author_new_old_map[node] == 2184:
            print(node)
            print(np.nonzero(author_term_mat[node]))
            print(author_term_mat[node][np.nonzero(author_term_mat[node])])

    fn = 'author_term_mat.pkl'
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(author_term_mat, wfile)

def relabeled_first_graph():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/DBLP/DBLP_Citation_2014_May/DM'

    # first graph, original id
    fn = 'dm_first_graph.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        first_dm_graph = pickle.load(rfile)

    print(first_dm_graph.number_of_nodes())

    author_old_new_map = {}
    author_new_old_map = {}
    for id, node in enumerate(first_dm_graph.nodes()):
        author_old_new_map[node] = id
        author_new_old_map[id] = node

    fn = 'dm_author_map.pkl'  # old id, new id
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump({'old_new': author_old_new_map, 'new_old': author_new_old_map}, wfile)

    print(len(author_new_old_map))
    print(author_new_old_map)

    string_map = {}
    for node in first_dm_graph.nodes():
        string_map[node] = '{}'.format(node)

    tmp_graph = nx.relabel_nodes(first_dm_graph, string_map)
    # print(string_map)

    string_old_new_map = {}
    for node in tmp_graph.nodes():  # tmp_graph, old id in string format
        # print(node)
        string_old_new_map[node] = author_old_new_map[int(node)]
        # print(node, author_old_new_map[int(node)])

    relabeled_dm_first_graph = nx.relabel_nodes(tmp_graph, string_old_new_map)

    fn = 'dm_relabeled_first_graph.pkl'
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(relabeled_dm_first_graph, wfile)

def calculate_coefficient():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/DBLP/DBLP_Citation_2014_May/DM'

    fn = 'author_term_mat.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        author_term_mat = pickle.load(rfile)

    print(len(author_term_mat))

    print([x for x in itertools.combinations([1, 2, 3], 2)])

    correlations = {}
    count = 0

    start_time = time.time()
    for u, v in itertools.combinations([i for i in range(len(author_term_mat))], 2):
        # print(u, v)
        # count += 1
        # if count == 1000000:
        #     print(time.time() - start_time)
        #     break
        print(count)
        x = pearsonr(author_term_mat[u], author_term_mat[v])

        y = np.dot(author_term_mat[u], author_term_mat[v])

        correlations[(u, v)] = x

        # mean_x = np.mean(author_term_mat[u])
        # mean_y = np.mean(author_term_mat[v])
        # std_x = np.std(author_term_mat[u])
        # std_y = np.std(author_term_mat[v])

        # print(mean_x, mean_y)
        # print(std_x, std_y)

        # x = np.dot(author_term_mat[u] - mean_x, author_term_mat[v] - mean_y) / (std_x * std_y) / len(author_term_mat[u])
        #
        # print(x)

        # break

    # fn = 'dm_correlations.pkl'
    # with open(os.path.join(path, fn), 'wb') as wfile:
    #     pickle.dump(correlations, wfile)

    # fn = 'dm_shrunk_correlations.pkl'
    # with open(os.path.join(path, fn), 'wb') as wfile:
    #     pickle.dump(correlations, wfile)


def construct_similarity_network():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/DBLP/DBLP_Citation_2014_May/DM'
    fn = 'dm_correlations.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        dm_correlations = pickle.load(rfile)

    # print(len(dm_correlations))


    pos_correlations = {}
    count = 0
    for key in dm_correlations.keys():
        if dm_correlations[key][0] > 0:
            count += 1
            print(count)
            pos_correlations[key] = dm_correlations[key][0]


    # sort pos_correlations, select top 30000
    sorted_x = sorted(pos_correlations.items(), key=operator.itemgetter(1))[::-1]
    # topk = 30000
    fn = 'dm_sorted_correlations.pkl'
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(sorted_x, wfile)


def construct_shrunk_similarity_network():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/DBLP/DBLP_Citation_2014_May/DM'
    fn = 'dm_correlations.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        dm_correlations = pickle.load(rfile)

    fn = 'dm_common_entry_num.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        common_entry_num = pickle.load(rfile)

    pos_correlations = {}
    lmbd = 5
    count = 0
    for key in dm_correlations.keys():
        if dm_correlations[key][0] > 0:
            count += 1
            print(count)
            if common_entry_num[key] == 0:
                pos_correlations[key] = 0.1/(5+0.1) * dm_correlations[key][0]
            else:
                pos_correlations[key] = common_entry_num[key] / float(common_entry_num[key] + lmbd) * dm_correlations[key][0]

    # sort pos_correlations, select top 30000
    sorted_x = sorted(pos_correlations.items(), key=operator.itemgetter(1))[::-1]
    # topk = 30000
    fn = 'dm_sorted_shrunk{}_correlations.pkl'.format(lmbd)
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(sorted_x, wfile)


def min_max_common_entries(): # 129, 0, 0.5020258226252269
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/DBLP/DBLP_Citation_2014_May/DM'

    fn = 'dm_author_term_mat.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        author_term_mat = pickle.load(rfile)

    print(len(author_term_mat))

    print([x for x in itertools.combinations([1, 2, 3], 2)])

    min_max = list()
    common_entry_num = {}
    for u, v in itertools.combinations([i for i in range(len(author_term_mat))], 2):
        x = np.multiply(author_term_mat[u], author_term_mat[v])
        z = np.count_nonzero(x)
        min_max.append(z)
        common_entry_num[(u, v)] = z

    print(np.max(list(min_max)))
    print(np.min(list(min_max)))
    print(np.mean(min_max))

    fn = 'dm_common_entry_num.pkl'
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(common_entry_num, wfile)


def count_papers():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/DBLP/DBLP_Citation_2014_May/DM'
    fn = 'dm_authors.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        db_authors = pickle.load(rfile)

    author_paper_count = np.zeros(len(db_authors))

    fn = 'dm_author_map.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        db_author_map = pickle.load(rfile)

    author_new_old_map = db_author_map['new_old']
    author_old_new_map = db_author_map['old_new']

    print(author_old_new_map)

    print(db_authors)

    fn = 'dm.txt'
    with open(os.path.join(path, fn)) as rfile:
        while True:
            record = []
            line = rfile.readline()
            if line.startswith('#*'):
                title = line
                record.append(line)
                while True:
                    line = rfile.readline()
                    record.append(line)

                    if line.startswith('#@'):
                        paper_authors = line.strip()[2:].split(',')
                        for author in paper_authors:
                            # print(author)
                            # print(db_authors[author])
                            if db_authors[author] not in author_old_new_map.keys():
                                continue
                            else:
                                # print(author_old_new_map[db_authors[author]])
                                author_paper_count[author_old_new_map[db_authors[author]]] += 1

                    if line.startswith('#!'):
                        break

            if not line:
                break

    print(author_paper_count[1260])
    print(author_new_old_map[1260])
    for k, v in db_authors.items():
        if v == author_new_old_map[1260]:
            print(k)

    fn = 'dm_author_paper_count.pkl'
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(author_paper_count, wfile)


def topk_names():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/DBLP/DBLP_Citation_2014_May/DM'

    fn = 'dm_authors.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        db_authors = pickle.load(rfile)

    fn = 'dm_author_map.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        db_author_map = pickle.load(rfile)

    author_new_old_map = db_author_map['new_old']
    author_old_new_map = db_author_map['old_new']

    fn = 'dm_author_paper_count.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        author_paper_count = pickle.load(rfile)

    print(np.mean(author_paper_count))

    topk = author_paper_count.argsort()[-80:][::-1]
    for x in topk:

        for name in db_authors.keys():
            if db_authors[name] == author_new_old_map[x]:
                # new id, old id, name
                print(x, author_new_old_map[x], name, author_paper_count[x])

def check_similarity_shrunk_normal():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/DBLP/DBLP_Citation_2014_May/DM'

    lmbd = 5
    topk = 30000
    fn = 'dm_sorted_shrunk{}_correlations.pkl'.format(topk, lmbd)
    with open(os.path.join(path, fn), 'rb') as rfile:
        sorted_x = pickle.load(rfile)[::-1][:3000]

    print(sorted_x)

    fn = 'dm_sorted_correlations.pkl'.format(topk)
    with open(os.path.join(path, fn), 'rb') as rfile:
        sorted_x = pickle.load(rfile)[::-1][:3000]

    print(sorted_x)

def generate_datasets():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/DBLP/DBLP_Citation_2014_May/DM'
    fn = 'dm_sorted_correlations.pkl'
    # lmbd = 5
    # fn = 'db_sorted_shrunk{}_correlations.pkl'.format(lmbd)
    with open(os.path.join(path, fn), 'rb') as rfile:
        db_correlations = pickle.load(rfile)

    fn = 'dm_relabeled_first_graph.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        db_first_graph = pickle.load(rfile)

    topk = 30000
    db_correlations = db_correlations[:topk]
    second_graph = nx.Graph()
    second_graph_edge_weight = {}
    second_graph.add_nodes_from([node for node in db_first_graph.nodes()])
    for x in db_correlations:
        u, v = x[0]
        cor_val = x[1]
        second_graph.add_edge(u, v)
        second_graph_edge_weight[(u, v)] = cor_val

    # print([node for node in db_first_graph.nodes()])

    fn = 'dm_author_paper_count.pkl' # db_author_paper_count include author not in graph, which are not counted, their values are zeros
    with open(os.path.join(path, fn), 'rb') as rfile:
        db_author_paper_count = pickle.load(rfile)

    weight = db_author_paper_count[:db_first_graph.number_of_nodes()]

    print(db_author_paper_count.shape)
    print(list(db_author_paper_count))
    print(np.nonzero(db_author_paper_count))

    fn = 'dm_top{}_dataset.pkl'.format(topk)
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump({'first_graph': db_first_graph, 'second_graph': second_graph, 'weight': weight, 'second_graph_edge_weight': second_graph_edge_weight}, wfile)


def get_stat_datasets():
    # path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/DBLP/DBLP_Citation_2014_May/DM'
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/DBLP/DBLP_Citation_2014_May/DB'
    # fn = 'dm_top30000_dataset.pkl'
    fn = 'db_top30000_dataset.pkl'
    with open(os.path.join(path, fn), 'rb') as rfile:
        dataset = pickle.load(rfile)

    first_graph = dataset['first_graph']
    second_graph = dataset['second_graph']
    print(nx.density(first_graph))
    print(nx.density(second_graph))

    print(first_graph.number_of_nodes())
    print(first_graph.number_of_edges())
    print(second_graph.number_of_nodes())
    print(second_graph.number_of_edges())
    print(nx.is_connected(first_graph))
    print(nx.is_connected(second_graph))

    lcc = max(nx.connected_component_subgraphs(second_graph), key=len)
    subgraph = nx.subgraph(second_graph, lcc)
    print(subgraph.number_of_nodes())
    print(subgraph.number_of_edges())

    # x = nx.shortest_path_length(second_graph, source=0)


    # for node in second_graph.nodes():
    #     if node not in x.keys():
    #         print(node)

    # print(x)


def generate_dm_dcs():
    rpath = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/DBLP/DBLP_Citation_2014_May/DM'
    fn = 'dm_top30000_dataset.pkl'
    with open(os.path.join(rpath, fn), 'rb') as rfile:
        instance = pickle.load(rfile)

    first_graph = instance['first_graph']
    second_graph = instance['second_graph']

    wpath = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/DBLP/DBLP_Citation_2014_May/dcs_dm'

    if first_graph.number_of_nodes() != second_graph.number_of_nodes():
        print('error !!!')

    fn = '01Nodes.txt'
    sorted_nodes = [node for node in first_graph.nodes()]
    with open(os.path.join(wpath, fn), 'w') as wfile:
        for node in sorted_nodes:
            wfile.write('{}\n'.format(node))

    fn = '02EdgesP.txt'
    with open(os.path.join(wpath, fn), 'w') as wfile:
        for edge in first_graph.edges():
            wfile.write('{} {}\n'.format(edge[0], edge[1]))

    fn = '03EdgesC.txt'
    with open(os.path.join(wpath, fn), 'w') as wfile:
        for edge in second_graph.edges():
            wfile.write('{} {} 1.0\n'.format(edge[0], edge[1]))

    weight = instance['weight']
    fn = 'sorted_attributes.txt'
    with open(os.path.join(wpath, fn), 'w') as wfile:
        x = np.argsort(weight)[::-1]
        for i in x:
            wfile.write('{} {:.5f}\n'.format(i, weight[i]))



if __name__ == '__main__':
    pass
    # extract_info_file() # extract papers in dm conferences, save to dm.txt, extract authors and assign each author with a id

    # construct_coauthor_network() # construct coauthorship network and extract largest connected component, node use original id

    # word_list() # extract word in titles and assign each word with a id from zero

    # construct_author_term_matrix() # construct dictionary, key, original author id, term frequency vector

    # test()

    # convert_id_from_zero()

    # calculate_coefficient()

    # construct_similarity_network()

    # min_max_common_entries()

    # count_papers()

    # relabeled_first_graph()

    # construct_similarity_network()

    # construct_shrunk_similarity_network()

    # check_similarity_shrunk_normal()

    # generate_datasets()

    # get_stat_datasets()
    # topk_names()

    # generate_dm_dcs()