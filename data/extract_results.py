#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : extract_results
# @Date     : 05/02/2019 18:15:21
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


from __future__ import print_function

import os
import sys

import numpy as np


def extract_ghtp_evo_bwsn(path=None):

    if not path:
        path = '/network/rit/lab/ceashpc/share_data/GraphOpt/log/ghtp/evo_bwsn'
    sparsity_list = [800, 850, 900, 950, 1000]
    # sparsity_list = [950]
    trade_off_list = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
    # trade_off_list = [0.5]
    learning_rate_list = [1., 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    # learning_rate_list = [0.1]

    results = {}
    for sparsity in sparsity_list:
        for trade_off in trade_off_list:
            for learning_rate in learning_rate_list:
                fn = 'bwsn_sparsity_{}_trade_{}_lr_{}.txt'.format(sparsity, trade_off, learning_rate)
                with open(os.path.join(path, fn)) as rfile:
                    for line in rfile:
                        if not line.startswith('av'):
                            continue
                        if line.startswith('average presision'):
                            terms = line.strip().split(':')
                            precision = float(terms[1].strip())
                            continue
                        if line.startswith('average recall'):
                            terms = line.strip().split(':')
                            recall = float(terms[1].strip())
                            continue
                        if line.startswith('average f-measure'):
                            terms = line.strip().split(':')
                            fm = float(terms[1].strip())
                            continue
                        if line.startswith('average iou'):
                            terms = line.strip().split(':')
                            iou = float(terms[1].strip())
                            continue
                        if line.startswith('avg refined prec'):
                            terms = line.strip().split(':')
                            refined_prec = float(terms[1].strip())
                            continue
                        if line.startswith('avg refined rec'):
                            terms = line.strip().split(':')
                            refined_rec = float(terms[1].strip())
                            continue
                        if line.startswith('avg refined fm'):
                            terms = line.strip().split(':')
                            refined_fm = float(terms[1].strip())
                            continue
                        if line.startswith('avg refined iou'):
                            terms = line.strip().split(':')
                            refined_iou = float(terms[1].strip())
                            continue
                        if line.startswith('average run time'):
                            terms = line.strip().split(':')
                            run_time = float(terms[1].strip())
                            continue

                results[(sparsity, trade_off, learning_rate)] = (precision, recall, fm, iou, refined_prec, refined_rec, refined_fm, refined_iou, run_time)

    # find best results
    best_refined_fm = 0.
    best_settings = []
    best_setting = None
    for k, v in results.iteritems():
        if v[6] > best_refined_fm:
            best_refined_fm = v[6]
            best_setting = k

    for k, v in results.iteritems():
        if v[6] >= best_refined_fm:
            # best_refined_fm = v[6]
            best_settings.append(k)

    print('best settings')
    print('number of best settings: {}'.format(len(best_settings)))
    print('sparsity     : {}'.format(best_settings[0][0]))
    print('trade_off    : {}'.format(best_settings[0][1]))
    print('learning_rate: {}'.format(best_settings[0][2]))
    print('\t'.join(['{:.5f}'.format(result) for result in results[best_settings[0]]]))


    # find shortest run_time
    run_time = np.inf
    best_setting = None
    for setting in best_settings:
        print(setting, results[setting])
        if results[setting][-1] < run_time:
            best_setting = setting
            run_time = results[setting][-1]

    print('sparsity     : {}'.format(best_setting[0]))
    print('trade_off    : {}'.format(best_setting[1]))
    print('learning_rate: {}'.format(best_setting[2]))
    print('\t'.join(['{:.5f}'.format(result) for result in results[best_setting]]))


def extract_ghtp_evo_dc():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/log/ghtp/evo_dc'
    sparsity = 300
    fn = 'sparsity={}.txt'.format(sparsity)

    results = {}
    with open(os.path.join(path, fn)) as rfile:
        while True:
            line = rfile.readline()
            if not line:
                break
            if 'setting' in line:
                while True:
                    line = rfile.readline()

                    if line.startswith('sparsity'):
                        terms = line.strip().split(':')
                        sparsity = int(terms[1].strip())
                        continue
                    if line.startswith('learning rate'):
                        terms = line.strip().split(':')
                        learning_rate = float(terms[1].strip())
                        continue
                    if line.startswith('trade off'):
                        terms = line.strip().split(':')
                        trade_off = float(terms[1].strip())
                        continue

                    if not line.startswith('av'):
                        continue

                    if line.startswith('average presision'):
                        terms = line.strip().split(':')
                        precision = float(terms[1].strip())
                        continue
                    if line.startswith('average recall'):
                        terms = line.strip().split(':')
                        recall = float(terms[1].strip())
                        continue
                    if line.startswith('average f-measure'):
                        terms = line.strip().split(':')
                        fm = float(terms[1].strip())
                        continue
                    if line.startswith('average iou'):
                        terms = line.strip().split(':')
                        iou = float(terms[1].strip())
                        continue
                    if line.startswith('avg refined prec'):
                        terms = line.strip().split(':')
                        refined_prec = float(terms[1].strip())
                        continue
                    if line.startswith('avg refined rec'):
                        terms = line.strip().split(':')
                        refined_rec = float(terms[1].strip())
                        continue
                    if line.startswith('avg refined fm'):
                        terms = line.strip().split(':')
                        refined_fm = float(terms[1].strip())
                        continue
                    if line.startswith('avg refined iou'):
                        terms = line.strip().split(':')
                        refined_iou = float(terms[1].strip())
                        continue
                    if line.startswith('average run time'):
                        terms = line.strip().split(':')
                        run_time = float(terms[1].strip())
                        print(sparsity, trade_off, learning_rate)
                        break

                results[(sparsity, trade_off, learning_rate)] = (precision, recall, fm, iou, refined_prec, refined_rec, refined_fm, refined_iou, run_time)

    # find best results
    best_refined_fm = 0.
    best_settings = []
    best_setting = None


    print(len(results))
    for k, v in results.iteritems():
        if v[6] > best_refined_fm:
            best_refined_fm = v[6]
            best_setting = k

    for k, v in results.iteritems():
        if v[6] >= best_refined_fm:
            best_settings.append(k)

    print('best settings')
    print('number of best settings: {}'.format(len(best_settings)))
    print('sparsity     : {}'.format(best_settings[0][0]))
    print('trade_off    : {}'.format(best_settings[0][1]))
    print('learning_rate: {}'.format(best_settings[0][2]))
    print('\t'.join(['{:.5f}'.format(result) for result in results[best_settings[0]]]))

    # find shortest run_time
    run_time = np.inf
    best_setting = None
    for setting in best_settings:
        print(setting, results[setting])
        if results[setting][-1] < run_time:
            best_setting = setting
            run_time = results[setting][-1]

    print('sparsity     : {}'.format(best_setting[0]))
    print('trade_off    : {}'.format(best_setting[1]))
    print('learning_rate: {}'.format(best_setting[2]))
    print('\t'.join(['{:.5f}'.format(result) for result in results[best_setting]]))


def extract_ghtp_evo_syn():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/log/ghtp/evo_syn'
    fn_tmp = 'syn_mu_{}_sparsity_{}_trade_{}_lr_{}.txt'

    results = {}

    sparsity_list = [100, 125, 150, 175, 200]
    # sparsity_list = [950]
    trade_off_list = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
    # trade_off_list = [0.5]
    learning_rate_list = [1., 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    # learning_rate_list = [0.1]

    mu = 5
    results = {}
    for sparsity in sparsity_list:
        for trade_off in trade_off_list:
            for learning_rate in learning_rate_list:
                fn = fn_tmp.format(mu, sparsity, trade_off, learning_rate)
                with open(os.path.join(path, fn)) as rfile:
                    while True:
                        line = rfile.readline()
                        if not line:
                            break
                        if 'setting' in line:
                            while True:
                                line = rfile.readline()

                                if line.startswith('sparsity'):
                                    terms = line.strip().split(':')
                                    sparsity = int(terms[1].strip())
                                    continue
                                if line.startswith('learning rate'):
                                    terms = line.strip().split(':')
                                    learning_rate = float(terms[1].strip())
                                    continue
                                if line.startswith('trade off'):
                                    terms = line.strip().split(':')
                                    trade_off = float(terms[1].strip())
                                    continue

                                if not line.startswith('av'):
                                    continue

                                if line.startswith('average presision'):
                                    terms = line.strip().split(':')
                                    precision = float(terms[1].strip())
                                    continue
                                if line.startswith('average recall'):
                                    terms = line.strip().split(':')
                                    recall = float(terms[1].strip())
                                    continue
                                if line.startswith('average f-measure'):
                                    terms = line.strip().split(':')
                                    fm = float(terms[1].strip())
                                    continue
                                if line.startswith('average iou'):
                                    terms = line.strip().split(':')
                                    iou = float(terms[1].strip())
                                    continue
                                if line.startswith('avg refined prec'):
                                    terms = line.strip().split(':')
                                    refined_prec = float(terms[1].strip())
                                    continue
                                if line.startswith('avg refined rec'):
                                    terms = line.strip().split(':')
                                    refined_rec = float(terms[1].strip())
                                    continue
                                if line.startswith('avg refined fm'):
                                    terms = line.strip().split(':')
                                    refined_fm = float(terms[1].strip())
                                    continue
                                if line.startswith('avg refined iou'):
                                    terms = line.strip().split(':')
                                    refined_iou = float(terms[1].strip())
                                    continue
                                if line.startswith('average run time'):
                                    terms = line.strip().split(':')
                                    run_time = float(terms[1].strip())
                                    print(sparsity, trade_off, learning_rate)
                                    break

                            results[(sparsity, trade_off, learning_rate)] = (precision, recall, fm, iou, refined_prec, refined_rec, refined_fm, refined_iou, run_time)

    # find best results
    best_refined_fm = 0.
    best_settings = []
    best_setting = None

    print(len(results))
    for k, v in results.iteritems():
        if v[6] > best_refined_fm:
            best_refined_fm = v[6]
            best_setting = k

    for k, v in results.iteritems():
        if v[6] >= best_refined_fm:
            best_settings.append(k)

    print('best settings')
    print('number of best settings: {}'.format(len(best_settings)))
    print('sparsity     : {}'.format(best_settings[0][0]))
    print('trade_off    : {}'.format(best_settings[0][1]))
    print('learning_rate: {}'.format(best_settings[0][2]))
    print('\t'.join(['{:.5f}'.format(result) for result in results[best_settings[0]]]))

    # find shortest run_time
    run_time = np.inf
    best_setting = None
    for setting in best_settings:
        print(setting, results[setting])
        if results[setting][-1] < run_time:
            best_setting = setting
            run_time = results[setting][-1]

    print('sparsity     : {}'.format(best_setting[0]))
    print('trade_off    : {}'.format(best_setting[1]))
    print('learning_rate: {}'.format(best_setting[2]))
    print('\t'.join(['{:.5f}'.format(result) for result in results[best_setting]]))


def extrac_iht_evo_bwsn():

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/log/iht/evo_bwsn'
    extract_ghtp_evo_bwsn(path)


def extract_node_run_time():

    path = '/network/rit/lab/ceashpc/share_data/DBLP/tmp'
    sparsity = 500
    num_blocks = 10
    normalized = True
    fn = 'node_{}_{}_{}.out'.format(sparsity, num_blocks, normalized)

    results = []
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            terms = line.strip().split(',')

            run_time = float(terms[3])
            prec = float(terms[8])
            rec = float(terms[9])
            fm = float(terms[10])
            iou = float(terms[11])
            results.append((prec, rec, fm, iou, run_time))

    results = np.array(results)
    for x in np.mean(results, axis=0):
        print(x, end='\t')


def extract_bj_run_time():
    path = '/network/rit/lab/ceashpc/share_data/DBLP/tmp'
    fn = 'beijing_290_100_True.out'
    results = []
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            terms = line.split(',')
            run_time = float(terms[3])
            prec = float(terms[8])
            rec = float(terms[9])
            fm = float(terms[10])
            iou = float(terms[11])
            results.append((prec, rec, fm, iou, run_time))

    results = np.array(results)
    print(results.shape)
    for x in np.mean(results, axis=0):
        print(x, end='\t')


def extract_cm_run_time():
    path = '/network/rit/lab/ceashpc/share_data/DBLP/tmp'
    fn = 'condmat_500_20_True.out'
    results = []
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            terms = line.split(',')
            run_time = float(terms[3])
            prec = float(terms[8])
            rec = float(terms[9])
            fm = float(terms[10])
            iou = float(terms[11])
            results.append((prec, rec, fm, iou, run_time))

    results = np.array(results)
    print(results.shape)
    for x in np.mean(results, axis=0):
        print(x, end='\t')

def extract_wv_run_time():
    path = '/network/rit/lab/ceashpc/share_data/DBLP/tmp'
    fn = 'wiki_150_20_True.out'
    results = []
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            terms = line.split(',')
            run_time = float(terms[3])
            prec = float(terms[8])
            rec = float(terms[9])
            fm = float(terms[10])
            iou = float(terms[11])
            results.append((prec, rec, fm, iou, run_time))

    results = np.array(results)
    print(results.shape)
    for x in np.mean(results, axis=0):
        print(x, end='\t')


def extract_dblp():

    path = '/network/rit/lab/ceashpc/share_data/DBLP/tmp'
    results = []

    for case_id in range(1):
        # fn = 'parallel_1995_2005_1500_2_100_20000_False_m2std_{}_new.out'.format(case_id)
        fn = 'tmp.txt'
        with open(os.path.join(path, fn)) as rfile:
            for line in rfile:
                print(fn, line.strip())
                terms = line.split(',')
                run_time = float(terms[3])
                prec = float(terms[8])
                rec = float(terms[9])
                fm = float(terms[10])
                iou = float(terms[11])
                results.append((prec, rec, fm, iou, run_time))

    results = np.array(results)
    print(results)
    print(results.shape)
    print(np.mean(results, axis=0))
    # for x in np.mean(results, axis=0):
    #     print(x, end='\t')

def extract_block_condmat():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/log/ghtp/block_condmat'
    results = []
    sparsity_list = [400, 500, 600]
    trade_off_list = [0.001, 0.0001]
    learning_rate = 1.0
    run_type = 'parallel'
    for sparsity in sparsity_list:
        for trade_off in trade_off_list:
            fn = 'condmat_sparsity_{}_trade_{}_lr_{}_{}.txt'.format(sparsity, trade_off, learning_rate, run_type)
            with open(os.path.join(path, fn)) as rfile:
                for line in rfile:
                    if line.startswith('average presision'):
                        terms = line.strip().split(':')
                        precision = float(terms[1].strip())
                        continue
                    if line.startswith('average recall'):
                        terms = line.strip().split(':')
                        recall = float(terms[1].strip())
                        continue
                    if line.startswith('average f-measure'):
                        terms = line.strip().split(':')
                        fm = float(terms[1].strip())
                        continue
                    if line.startswith('average iou'):
                        terms = line.strip().split(':')
                        iou = float(terms[1].strip())
                        continue
                    if line.startswith('avg refined prec'):
                        terms = line.strip().split(':')
                        refined_prec = float(terms[1].strip())
                        continue
                    if line.startswith('avg refined rec'):
                        terms = line.strip().split(':')
                        refined_rec = float(terms[1].strip())
                        continue
                    if line.startswith('avg refined fm'):
                        terms = line.strip().split(':')
                        refined_fm = float(terms[1].strip())
                        continue
                    if line.startswith('avg refined iou'):
                        terms = line.strip().split(':')
                        refined_iou = float(terms[1].strip())
                        continue
                    if line.startswith('average run time'):
                        terms = line.strip().split(':')
                        run_time = float(terms[1].strip())
                        print(sparsity, trade_off, learning_rate)
                        break

            print(sparsity, trade_off, precision, recall, fm, iou, refined_prec, refined_rec, refined_fm, refined_iou, run_time)
            # results[(sparsity, trade_off, learning_rate)] = (precision, recall, fm, iou, refined_prec, refined_rec, refined_fm, refined_iou, run_time)

def extract_ba():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/log/ghtp/block_ba'
    # deg = 3
    lr = 1.0
    nodes = 100000
    normalize = False
    run_type = 'parallel'
    # run_type = 'serial'
    for i in range(1, 11):
        nodes = 1000 * i
        # deg = i
        deg = 3
        # sparsity = int(nodes * 0.005)
        sparsity = int(nodes * 0.05)
        for trade_off in [0.001, 0.0001]:
            if normalize:
                fn = 'ba_nodes_{}_deg_{}_sparsity_{}_trade_{}_lr_{}_{}_{}.txt'.format(nodes, deg, sparsity, trade_off, lr, normalize, run_type)
            else:
                fn = 'ba_nodes_{}_deg_{}_sparsity_{}_trade_{}_lr_{}_{}.txt'.format(nodes, deg, sparsity, trade_off, lr, run_type)
            with open(os.path.join(path, fn)) as rfile:
                for line in rfile:
                    if line.startswith('average f-measure'):
                        terms = line.strip().split(':')
                        fm = float(terms[1].strip())
                        continue
                    if line.startswith('average run time'):
                        terms = line.strip().split(':')
                        run_time = float(terms[1].strip())
                        break

            print(nodes, deg, sparsity, trade_off, run_time, fm)

def extract_wikivote():

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/log/ghtp/block_wikivote'
    results = []
    sparsity_list = [160, 180, 200]
    trade_off_list = [0.001, 0.0001]
    learning_rate = 1.0
    # run_type = 'parallel'
    run_type = 'serial'
    for sparsity in sparsity_list:
        for trade_off in trade_off_list:
            fn = 'wikivote_sparsity_{}_trade_{}_lr_{}_{}.txt'.format(sparsity, trade_off, learning_rate, run_type)
            with open(os.path.join(path, fn)) as rfile:
                for line in rfile:
                    if line.startswith('average presision'):
                        terms = line.strip().split(':')
                        precision = float(terms[1].strip())
                        continue
                    if line.startswith('average recall'):
                        terms = line.strip().split(':')
                        recall = float(terms[1].strip())
                        continue
                    if line.startswith('average f-measure'):
                        terms = line.strip().split(':')
                        fm = float(terms[1].strip())
                        continue
                    if line.startswith('average iou'):
                        terms = line.strip().split(':')
                        iou = float(terms[1].strip())
                        continue
                    if line.startswith('avg refined prec'):
                        terms = line.strip().split(':')
                        refined_prec = float(terms[1].strip())
                        continue
                    if line.startswith('avg refined rec'):
                        terms = line.strip().split(':')
                        refined_rec = float(terms[1].strip())
                        continue
                    if line.startswith('avg refined fm'):
                        terms = line.strip().split(':')
                        refined_fm = float(terms[1].strip())
                        continue
                    if line.startswith('avg refined iou'):
                        terms = line.strip().split(':')
                        refined_iou = float(terms[1].strip())
                        continue
                    if line.startswith('average run time'):
                        terms = line.strip().split(':')
                        run_time = float(terms[1].strip())
                        print(sparsity, trade_off, learning_rate)
                        break

            print(sparsity, trade_off, precision, recall, fm, iou, refined_prec, refined_rec, refined_fm, refined_iou, run_time)


def extract_bj():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/log/ghtp/block_bj'
    results = []
    sparsity_list = [280, 300, 320]
    # trade_off_list = [0.001, 0.0005, 0.0001]
    trade_off_list = [0.001, 0.0001]
    learning_rate = 1.0
    run_type = 'parallel'
    # run_type = 'serial'
    normalize = True
    for sparsity in sparsity_list:
        for trade_off in trade_off_list:
            if normalize:
                fn = 'bj_sparsity_{}_trade_{}_lr_{}_{}_{}.txt'.format(sparsity, trade_off, learning_rate, normalize, run_type)
            else:
                fn = 'bj_sparsity_{}_trade_{}_lr_{}_{}.txt'.format(sparsity, trade_off, learning_rate, run_type)
            with open(os.path.join(path, fn)) as rfile:
                for line in rfile:
                    if line.startswith('average presision'):
                        terms = line.strip().split(':')
                        precision = float(terms[1].strip())
                        continue
                    if line.startswith('average recall'):
                        terms = line.strip().split(':')
                        recall = float(terms[1].strip())
                        continue
                    if line.startswith('average f-measure'):
                        terms = line.strip().split(':')
                        fm = float(terms[1].strip())
                        continue
                    if line.startswith('average iou'):
                        terms = line.strip().split(':')
                        iou = float(terms[1].strip())
                        continue
                    if line.startswith('avg refined prec'):
                        terms = line.strip().split(':')
                        refined_prec = float(terms[1].strip())
                        continue
                    if line.startswith('avg refined rec'):
                        terms = line.strip().split(':')
                        refined_rec = float(terms[1].strip())
                        continue
                    if line.startswith('avg refined fm'):
                        terms = line.strip().split(':')
                        refined_fm = float(terms[1].strip())
                        continue
                    if line.startswith('avg refined iou'):
                        terms = line.strip().split(':')
                        refined_iou = float(terms[1].strip())
                        continue
                    if line.startswith('average run time'):
                        terms = line.strip().split(':')
                        run_time = float(terms[1].strip())
                        print(sparsity, trade_off, learning_rate)
                        break

            print(sparsity, trade_off, precision, recall, fm, iou, refined_prec, refined_rec, refined_fm, refined_iou, run_time)


def extract(fn):

    precision = recall = fm = iou = refined_prec = refined_rec = refined_fm = refined_iou = run_time = 0.
    with open(fn) as rfile:
        for line in rfile:
                    if line.startswith('average presision'):
                        terms = line.strip().split(':')
                        precision = float(terms[1].strip())
                        continue
                    if line.startswith('average recall'):
                        terms = line.strip().split(':')
                        recall = float(terms[1].strip())
                        continue
                    if line.startswith('average f-measure'):
                        terms = line.strip().split(':')
                        fm = float(terms[1].strip())
                        continue
                    if line.startswith('average iou'):
                        terms = line.strip().split(':')
                        iou = float(terms[1].strip())
                        continue
                    if line.startswith('avg refined prec'):
                        terms = line.strip().split(':')
                        refined_prec = float(terms[1].strip())
                        continue
                    if line.startswith('avg refined rec'):
                        terms = line.strip().split(':')
                        refined_rec = float(terms[1].strip())
                        continue
                    if line.startswith('avg refined fm'):
                        terms = line.strip().split(':')
                        refined_fm = float(terms[1].strip())
                        continue
                    if line.startswith('avg refined iou'):
                        terms = line.strip().split(':')
                        refined_iou = float(terms[1].strip())
                        continue
                    if line.startswith('average run time'):
                        terms = line.strip().split(':')
                        run_time = float(terms[1].strip())
                        break


    return precision, recall, fm, iou, refined_prec, refined_rec, refined_fm, refined_iou, run_time


def extract_evo_dc():

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/log/iht/evo_dc'

    best_fm = 0.
    best_setting = None

    for fn in os.listdir(path):
        terms = fn[:fn.find('.txt')].split('_')
        sparsity = terms[2]
        trade_off = terms[4]
        lr = terms[6]
        precision, recall, fm, iou, refined_prec, refined_rec, refined_fm, refined_iou, run_time = extract(os.path.join(path, fn))

        print(sparsity, trade_off, lr, precision, recall, fm, iou, refined_prec, refined_rec, refined_fm, refined_iou, run_time)

        if best_fm < refined_fm:
            best_fm = refined_fm
            best_setting = (sparsity, trade_off, lr)

    print(best_setting)

def extract_evo_bj():

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/log/iht/evo_bj'

    best_fm = 0.
    best_setting = None

    for fn in os.listdir(path):
        terms = fn[:fn.find('.txt')].split('_')
        sparsity = terms[2]
        trade_off = terms[4]
        lr = terms[6]
        precision, recall, fm, iou, refined_prec, refined_rec, refined_fm, refined_iou, run_time = extract(os.path.join(path, fn))

        print(sparsity, trade_off, lr, precision, recall, fm, iou, refined_prec, refined_rec, refined_fm, refined_iou, run_time)

        if best_fm < refined_fm:
            best_fm = refined_fm
            best_setting = (sparsity, trade_off, lr)

    print(best_setting)

def extract_evo_syn():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/log/iht/evo_syn'

    best_fm = 0.
    best_setting = None

    for fn in os.listdir(path):
        terms = fn[:fn.find('.txt')].split('_')
        mu = terms[2]
        sparsity = terms[4]
        trade_off = terms[6]
        lr = terms[8]
        precision, recall, fm, iou, refined_prec, refined_rec, refined_fm, refined_iou, run_time = extract(os.path.join(path, fn))

        print(mu, sparsity, trade_off, lr, precision, recall, fm, iou, refined_prec, refined_rec, refined_fm, refined_iou,
              run_time)

        if best_fm < refined_fm:
            best_fm = refined_fm
            best_setting = (sparsity, trade_off, lr)

    print(best_setting)


def extract_block_bj():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/log/ghtp/block_bj'

    best_fm = 0.
    best_setting = None

    for fn in os.listdir(path):
        terms = fn[:fn.find('.txt')].split('_')
        sparsity = terms[2]
        trade_off = terms[4]
        lr = terms[6]
        run_type = terms[7]
        precision, recall, fm, iou, refined_prec, refined_rec, refined_fm, refined_iou, run_time = extract(os.path.join(path, fn))

        print(run_type, sparsity, trade_off, lr, precision, recall, fm, iou, refined_prec, refined_rec, refined_fm, refined_iou, run_time)

        if best_fm < refined_fm:
            best_fm = refined_fm
            best_setting = (sparsity, trade_off, lr)

    print(best_setting)


def extract_cw_file(fn):

    results = []
    with open(fn) as rfile:
        for line in rfile:
            terms = line.strip().split(',')
            trade_off = float(terms[0])
            sparsity = int(terms[1])
            case_id = int(terms[2])
            run_time = float(terms[3])
            prec = float(terms[8])
            rec = float(terms[9])
            fm = float(terms[10])
            iou = float(terms[11])

            results.append((prec, rec, fm, iou, run_time))

    results = np.array(results)

    mean_results = np.mean(results, axis=0)
    print('sparsity={}, trade_off={}'.format(sparsity, trade_off))
    print(mean_results.reshape((-1, 1)))

def extract_ba_serial_new():
    num_nodes = 10000
    num_blocks = 10
    deg = 3
    sparsity = int(num_nodes / num_blocks / 2.)
    trade_off = 0.01
    run_type = 'serial'

    fn = '/network/rit/lab/ceashpc/share_data/GraphOpt/log/ghtp/block_ba_5/node_{}_{}_{}_{}_{}.out'.format(num_nodes, deg, sparsity, trade_off, run_type)

    extract_cw_file(fn)

def extract_ba_parallel_new():
    num_nodes = 10000
    num_blocks = 10
    deg = 3
    sparsity = int(num_nodes / num_blocks / 2.)
    trade_off = 0.0005
    run_type = 'parallel'

    fn = '/network/rit/lab/ceashpc/share_data/GraphOpt/log/ghtp/block_ba_5/node_{}_{}_{}_{}_{}.out'.format(num_nodes, deg, sparsity, trade_off, run_type)

    extract_cw_file(fn)


def extract_ba_serial_new():
    num_nodes = 100000
    num_blocks = 100
    deg = 10
    sparsity = int(num_nodes / num_blocks / 2.)
    trade_off = 0.0005
    run_type = 'parallel'

    fn = '/network/rit/lab/ceashpc/share_data/GraphOpt/log/ghtp/block_ba_6/nodes_{}_blocks_{}_mu_5_subsize_5000.0_5000.0_deg_{}_train_0_serial_2.out'.format(num_nodes, num_blocks, deg)

    extract_cw_file(fn)


def extract_ba_parallel_edge_block_4():
    num_nodes = 100000
    num_blocks = 100
    deg = 9
    run_type = 'parallel'

    fn = '/network/rit/lab/ceashpc/share_data/GraphOpt/log/ghtp/block_ba_4/nodes_{}_blocks_{}_mu_5_subsize_5000.0_5000.0_deg_{}_train_{}.out'.format(num_nodes, num_blocks, deg, run_type)

    extract_cw_file(fn)


def extract_bj_parallel():


    sparsity = 300
    trade_off = 0.0001

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/log/ghtp/block_bj_2'

    fn = 'parallel_{}_{}.out'.format(sparsity, trade_off)
    extract_cw_file(os.path.join(path, fn))

    # results = []
    # hour_list = [17, 18]
    # for hour in hour_list:
    #     for case_id in range(6):
    #         fn = 'parallel_{}_{}.out'.format(hour, case_id)
    #         with open(os.path.join(path, fn)) as rfile:
    #             for line in rfile:
    #                 terms = line.strip().split(',')
    #                 trade_off = float(terms[0])
    #                 sparsity = int(terms[1])
    #                 case_id = int(terms[2])
    #                 run_time = float(terms[3])
    #                 prec = float(terms[8])
    #                 rec = float(terms[9])
    #                 fm = float(terms[10])
    #                 iou = float(terms[11])
    #
    #             results.append((prec, rec, fm, iou, run_time))
    #
    # results = np.array(results)
    # print(results.shape)
    # print(results)
    #
    # mean_results = np.mean(results, axis=0)
    # print('sparsity={}, trade_off={}'.format(sparsity, trade_off))
    # print(mean_results.reshape((-1, 1)))


if __name__ == '__main__':

    pass

    # extract_ba_serial_new()

    extract_bj_parallel()

    # extract_ba_parallel_edge_block_4()

    # extract_ba_serial_new()

    # extract_ba_parallel_new()

    # extract_block_bj()

    # extract_evo_syn()

    # extract_evo_bj()

    # extract_evo_dc()

    # extract_bj()

    # extract_wikivote()

    # extract_ba()

    # extract_block_condmat()

    # extract_ghtp_evo_bwsn()

    # extract_ghtp_evo_dc()

    # extract_ghtp_evo_syn()

    # extrac_iht_evo_bwsn()

    # extract_node_run_time()


    # extract_bj_run_time()

    # extract_cm_run_time()

    # extract_wv_run_time()

    # extract_dblp()
