#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : extract_results
# @Date     : 08/04/2019 20:45:01
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :

from __future__ import print_function

import os
import sys
import pickle

import numpy as np
import pandas as pd


def extract_results():
    path = '/network/rit/lab/ceashpc/fjie/projects/GraphOpt/block_ghtp'
    # fn = 'slurm-236129.out'
    # fn = 'slurm-236238.out'
    # fn = 'slurm-236239.out'
    fn = 'slurm-236241.out'
    # fn = 'slurm-236242.out'
    # fn = 'slurm-236243.out'
    # fn = 'slurm-236244.out'
    # fn = 'slurm-236245.out'
    # fn = 'slurm-236246.out'
    # fn = 'slurm-236247.out'

    with open(os.path.join(path, fn)) as rfile:
        while True:
            line = rfile.readline()
            if 'performance' in line:
                while True:
                    line = rfile.readline()
                    if 'density' in line:
                        terms = line.split(':')
                        density = float(terms[1])
                        break
                    if 'precision' in line:
                        terms = line.split(':')
                        precision = float(terms[1])
                    if 'recall' in line:
                        terms = line.split(':')
                        recall = float(terms[1])
                    if 'f-measure' in line:
                        terms = line.split(':')
                        fm = float(terms[1])
                    if 'iou' in line:
                        terms = line.split(':')
                        iou = float(terms[1])

                print(precision, recall, fm, iou, density, sep='\t', end='\t')

            if not line:
                break



def extract_density_projection():

    path = '/network/rit/lab/ceashpc/fjie/projects/GraphOpt/block_ghtp'
    fn = 'slurm-256653.out'
    all_results = []
    with open(os.path.join(path, fn)) as rfile:
        while True:
            line = rfile.readline().strip()
            if not line:
                break

            if line.startswith('case'):
                case_id = int(line.split(':')[1])
                case_result = []
                while True:
                    line = rfile.readline().strip()
                    if line.startswith('-----') and 'performance' in line:
                        line = rfile.readline().strip()
                        prec = float(line.split(':')[1])
                        line = rfile.readline().strip()
                        rec = float(line.split(':')[1])
                        line = rfile.readline().strip()
                        fm = float(line.split(':')[1])
                        line = rfile.readline().strip()
                        iou = float(line.split(':')[1])
                        line = rfile.readline().strip()
                        density = float(line.split(':')[1])
                        case_result.append((prec, rec, fm, iou, density))
                        # print(prec, rec, fm, iou, density, sep='\n')

                    if len(case_result) == 8:
                        break


            all_results.append(case_result)

    all_results = np.array(all_results)
    print(all_results.shape)
    all_results = np.reshape(all_results, (10, 40))
    all_results = np.transpose(all_results)
    print(all_results)

    mean_results = np.mean(all_results, axis=1)

    np.set_printoptions(linewidth=np.inf, precision=5)

    np.savetxt(fname=sys.stdout, X=all_results, delimiter='|', newline='\n', fmt='% .5f')
    np.savetxt(fname=sys.stdout, X=mean_results, delimiter='|', newline='\n', fmt='% .5f')



    # results_tab = pd.DataFrame(all_results)
    # with pd.option_context('display.max_rows', 41, 'display.max_columns', 11):
    #     # pd.set_option('display.max_columns', 41)
    #     pd.set_option('display.max_colwidth', 2000)
    #     print(results_tab)


def extract_eventallpairs():
    path = '/network/rit/lab/ceashpc/fjie/projects/GraphOpt/references/dcs'
    fn = 'slurm-256702.out'
    all_results_1 = []
    all_results_2 = []
    with open(os.path.join(path, fn)) as rfile:
        while True:
            line = rfile.readline().strip()
            if not line:
                break

            if line.startswith('lambda'):
                lmbd = int(line.split(' ')[1])
                print(lmbd)
                while True:
                    line = rfile.readline().strip()
                    if line.startswith('mean ['):
                        terms = line[line.find('[')+1:line.find(']')].split(',')
                        results_1 = tuple([float(term) for term in terms])
                        all_results_1.append(results_1)

                    if line.startswith('mean_2 ['):
                        terms = line[line.find('[')+1:line.find(']')].split(',')
                        results_2 = tuple([float(term) for term in terms])
                        all_results_2.append(results_2)
                        break

    all_results_1 = np.array(all_results_1)
    all_results_2 = np.array(all_results_2)
    print(all_results_1)

    np.set_printoptions(linewidth=2000)
    print(all_results_1.shape)
    np.savetxt(fname=sys.stdout, X=all_results_1, delimiter='|', newline='\n', fmt='% .5f')
    print(all_results_2.shape)
    np.savetxt(fname=sys.stdout, X=all_results_2, delimiter='|', newline='\n', fmt='% .5f')


def extract_dcs():
    num_nodes = 1000
    first_deg = 3
    second_deg = 10
    subgraph_size = 100
    mu = 3
    restart = 0.3
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic'
    for case_id in range(1):
        fn = 'dual_mu_{}_nodes_{}_deg1_{}_deg2_{}_subsize_{}_restart_{}_{}.pkl'.format(mu, num_nodes, first_deg, second_deg, subgraph_size, restart, case_id)
        with open(os.path.join(path, fn), 'rb') as rfile:
            instance = pickle.load(rfile)

    first_graph = instance['first_graph']
    second_graph = instance['second_graph']
    # for node_1, node_2 in first_graph.edges():
    for node_1, node_2 in second_graph.edges():
        print(node_1, node_2)





    # case_id = 0
    # path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/synthetic/dcs/mu3_restart0.3/case_{}'.format(case_id)
    #
    # fn = '01DenseSubgraphNodes.txt'
    # subgraph = set()
    # with open(os.path.join(path, fn)) as rfile:
    #     for line in rfile:
    #         node = int(line.strip())
    #         subgraph.add(node)


if __name__ == '__main__':
    pass

    # extract_results()

    # extract_density_projection()

    # extract_eventallpairs()

    extract_dcs()