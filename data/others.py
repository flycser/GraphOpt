#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : others
# @Date     : 04/10/2019 16:45:16
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :

from __future__ import print_function

import os

import pickle

import numpy as np


def convert_keys_ijcai():
    # path = '/network/rit/lab/ceashpc/share_data/GraphOpt/ijcai/app2/BA'
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/ijcai/app2/CondMat'
    # fn = 'nodes_100000_blocks_100_mu_5_subsize_5000.0_5000.0_deg_10_train_0.pkl'
    fn = 'blocks_20_mu_5_true_1000_case_9.pkl'
    rfn = os.path.join(path, fn)
    with open(rfn, 'rb') as rfile:
        dataset = pickle.load(rfile)

    # instance = dataset[0]
    instance = dataset

    converted_instance = {}
    converted_dataset = []
    converted_instance['graph'] = instance['graph']
    converted_instance['subgraph'] = instance['true_subgraph']
    converted_instance['features'] = np.array(instance['features'])
    converted_instance['block_node_sets'] = instance['nodes_set']
    converted_instance['block_boundary_edges_dict'] = instance['block_boundary_edges_dict']

    converted_dataset.append(converted_instance)

    fn = 'test_9.pkl'
    wfn = os.path.join(path, fn)
    with open(wfn, 'wb') as wfile:
        pickle.dump(converted_dataset, wfile)


def convert_evo_syn_ijcai():

    mu = 3
    data_type = 'test'
    path = '/network/rit/lab/ceashpc/share_data/BA/run0/dgmp'
    fn = 'nodes_3000_windows_7_mu_{:d}_subsize_100_300_range_1_5_overlap_0.5_m_{}.pkl'.format(mu, data_type)
    rfn = os.path.join(path, fn)
    with open(rfn, 'rb') as rfile:
        dataset = pickle.load(rfile)

    converted_dataset = []
    for k in dataset.keys():
        print('instance: {:d}'.format(k))
        instance = dataset[k]
        converted_instance = {}
        converted_instance['graph'] = instance['graph']
        converted_instance['subgraphs'] = instance['true_subgraphs']
        converted_instance['features'] = instance['features']

        converted_dataset.append(converted_instance)


    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/app1/synthetic'
    wfn = os.path.join(path, fn)
    with open(wfn, 'wb') as wfile:
        pickle.dump(converted_dataset, wfile)

def convert_evo_dc_ijcai():
    # week_list = [0, 1] # train
    week_list = [2, 3] # test
    weekday_list = [0, 1, 2, 3, 4]
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/app1/dc'
    # top_k = 30
    # top_k = 5
    top_k = 10

    dataset = []
    for week in week_list:
        for weekday in weekday_list:
            fn = 'week_{:d}_weekday_{}.pkl'.format(week, weekday)
            rfn = os.path.join(path, fn)
            with open(rfn, 'rb') as rfile:
                instance = pickle.load(rfile)

            converted_instance = {}
            converted_instance['graph'] = instance['graph']
            converted_instance['features'] = np.array(instance['features'])
            top_k_true_subgraphs = instance['top_k_true_subgraphs'][:top_k]
            T = len(top_k_true_subgraphs[0])
            combined_true_subgraphs = [[] for i in range(T)]
            for true_subgraphs in top_k_true_subgraphs:
                for i in range(T):
                    combined_true_subgraphs[i] += true_subgraphs[i]
            converted_instance['subgraphs'] = combined_true_subgraphs
            dataset.append(converted_instance)

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/app1/dc'
    # fn = 'train.pkl'
    fn = 'test_10.pkl'
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(dataset, wfile)


def convert_evo_bwsn_ijcai():
    # noise_level = [0] # train
    noise_level = [2, 4, 6, 8, 10] # test
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/app1/bwsn'


    for noise in noise_level:
        dataset = []
        fn = 'Water-source-12500_noise_{}.pkl'.format(noise)
        rfn = os.path.join(path, fn)
        with open(rfn, 'rb') as rfile:
            instance = pickle.load(rfile)

        converted_instance = {}
        converted_instance['graph'] = instance['graph']
        converted_instance['features'] = np.array(instance['features'])
        true_subgraphs = instance['true_subgraphs']
        converted_instance['subgraphs'] = true_subgraphs
        dataset.append(converted_instance)

        # fn = 'train_noise_{}.pkl'.format(noise) # train
        fn = 'test_noise_{}.pkl'.format(noise) # test
        with open(os.path.join(path, fn), 'wb') as wfile:
            pickle.dump(dataset, wfile)


def convert_evo_bj_ijcai():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/app1/bj'
    # date = '20130912' # train
    date = '20130913' # test
    hour_list = range(5, 23)
    top_k = 30

    dataset = []
    for hour in hour_list:
        fn = '{}_hour_{}.pkl'.format(date, hour)
        with open(os.path.join(path, fn), 'rb') as rfile:
            instance = pickle.load(rfile)

        converted_instance = {}
        converted_instance['graph'] = instance['graph']
        converted_instance['features'] = np.array(instance['features'])
        top_k_true_subgraphs = instance['top_k_true_subgraphs'][:top_k]
        T = len(top_k_true_subgraphs[0])
        combined_true_subgraphs = [[] for i in range(T)]
        for true_subgraphs in top_k_true_subgraphs:
            for i in range(T):
                combined_true_subgraphs[i] += true_subgraphs[i]
        converted_instance['subgraphs'] = combined_true_subgraphs
        dataset.append(converted_instance)

    # fn = 'train.pkl'
    fn = 'test.pkl'
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(dataset, wfile)


def normalize_beijing():
    path = '/network/rit/lab/ceashpc/share_data/Beijing/run4/mgmp'
    fn = '20130912_hour_17_ind_0.pkl'

    with open(os.path.join(path, fn), 'rb') as rfile:
        dataset = pickle.load(rfile)

    print(dataset.keys())
    print(len(dataset['nodes_set']))
    print(dataset['graph'].number_of_nodes())
    print(dataset['graph'].number_of_edges())
    # dataset['features'] = dataset['features'] / np.max(dataset['features'])

    # path = '/network/rit/lab/ceashpc/share_data/DBLP/tmp'
    # fn = '20130912_hour_17_ind_0_normalized.pkl'
    # with open(os.path.join(path, fn), 'wb') as wfile:
    #     pickle.dump(dataset, wfile)


def normalize_wiki():

    for case_id in range(10):
        print(case_id)
        path = '/network/rit/lab/ceashpc/share_data/GraphOpt/ijcai/app2/Wikivote'
        fn = 'blocks_20_mu_5_true_1000_case_{}.pkl'.format(case_id)

        with open(os.path.join(path, fn), 'rb') as rfile:
            dataset = pickle.load(rfile)

        print(dataset.keys())
        print(len(dataset['nodes_set']))
        print(dataset['graph'].number_of_nodes())
        print(dataset['graph'].number_of_edges())
        dataset['features'] = dataset['features'] / np.max(dataset['features'])

        path = '/network/rit/lab/ceashpc/share_data/DBLP/wiki'
        fn = 'blocks_20_mu_5_true_1000_case_{}.pkl'.format(case_id)
        with open(os.path.join(path, fn), 'wb') as wfile:
            pickle.dump(dataset, wfile)


def normalize_condmat():

    for case_id in range(10):
        print(case_id)
        path = '/network/rit/lab/ceashpc/share_data/GraphOpt/ijcai/app2/CondMat'
        fn = 'blocks_20_mu_5_true_1000_case_{}.pkl'.format(case_id)

        with open(os.path.join(path, fn), 'rb') as rfile:
            dataset = pickle.load(rfile)

        print(dataset.keys())
        print(len(dataset['nodes_set']))
        print(dataset['graph'].number_of_nodes())
        print(dataset['graph'].number_of_edges())
        dataset['features'] = dataset['features'] / np.max(dataset['features'])

        path = '/network/rit/lab/ceashpc/share_data/DBLP/condmat'
        fn = 'blocks_20_mu_5_true_1000_case_{}.pkl'.format(case_id)
        with open(os.path.join(path, fn), 'wb') as wfile:
            pickle.dump(dataset, wfile)


def normalize_beijing():
    date = '20130913'
    hour_list = [17, 18]

    for hour in hour_list:
        for case_id in range(6):
            print(case_id)
            path = '/network/rit/lab/ceashpc/share_data/GraphOpt/ijcai/app2/Beijing'
            fn = '{}_hour_{}_ind_{}.pkl'.format(date, hour, case_id)

            with open(os.path.join(path, fn), 'rb') as rfile:
                dataset = pickle.load(rfile)

            print(dataset.keys())
            print(len(dataset['nodes_set']))
            print(dataset['graph'].number_of_nodes())
            print(dataset['graph'].number_of_edges())
            dataset['features'] = dataset['features'] / np.max(dataset['features'])

            path = '/network/rit/lab/ceashpc/share_data/DBLP/beijing'
            fn = '{}_hour_{}_ind_{}.pkl'.format(date, hour, case_id)
            with open(os.path.join(path, fn), 'wb') as wfile:
                pickle.dump(dataset, wfile)

def normalize_node():
    path_1 = '/network/rit/lab/ceashpc/share_data/BA/run5'
    path_2 = '/network/rit/lab/ceashpc/share_data/DBLP/node'
    for idx in range(1,11):
        for case_id in range(2):
            print(idx, case_id)
            fn = 'nodes_{}_blocks_10_mu_5_subsize_{:.1f}_{:.1f}_deg_3_train_{}.pkl'.format(1000*idx, 100.0*idx, 100.0*idx, case_id)
            with open(os.path.join(path_1, fn), 'rb') as rfile, open(os.path.join(path_2, fn), 'wb') as wfile:
                dataset = pickle.load(rfile)
                print(dataset.keys())
                dataset[0]['features'] = dataset[0]['features'] / np.max(dataset[0]['features'])
                dataset = dataset[0]

                pickle.dump(dataset, wfile)


def normalize_edge():
    path_1 = '/network/rit/lab/ceashpc/share_data/BA/run3'
    path_2 = '/network/rit/lab/ceashpc/share_data/DBLP/edge'
    for idx in range(3,11):
        for case_id in range(2):
            print(idx, case_id)
            fn = 'nodes_100000_blocks_100_mu_5_subsize_5000.0_5000.0_deg_{}_train_{}.pkl'.format(idx, case_id)
            with open(os.path.join(path_1, fn), 'rb') as rfile, open(os.path.join(path_2, fn), 'wb') as wfile:
                dataset = pickle.load(rfile)
                print(dataset.keys())
                dataset[0]['features'] = dataset[0]['features'] / np.max(dataset[0]['features'])
                dataset = dataset[0]

                pickle.dump(dataset, wfile)

def filter_pending_files():

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/log/ghtp/evo_bj'

    for fn in os.listdir(path):
        with open(os.path.join(path, fn)) as rfile:
            tag = False
            for line in rfile:
                if '----- average performance -----' in line:
                    tag = True
                    break

            if not tag:
                print(fn, end=' ')

def convert():

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/app1/synthetic'
    mu = 3
    data_type = 'test'

    fn = 'nodes_3000_windows_7_mu_{}_subsize_100_300_range_1_5_overlap_0.5_m_{}.pkl'.format(mu, data_type)
    with open(os.path.join(path, fn), 'rb') as rfile:
        dataset = pickle.load(rfile)

    new_dataset = []
    for i in range(len(dataset)):
        new_dataset.append(dataset[i])

    fn = '{}_mu_{}.pkl'.format(data_type, mu)
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(new_dataset, wfile)





if __name__ == '__main__':

    # convert()

    # convert_k eys_ijcai()

    # convert_evo_syn_ijcai()

    convert_evo_dc_ijcai()

    # convert_evo_bwsn_ijcai()


    # convert_evo_bj_ijcai()

    # normalize_beijing()

    # normalize_wiki()

    # normalize_condmat()

    # normalize_beijing()

    # normalize_node()

    # normalize_edge()

    # filter_pending_files()