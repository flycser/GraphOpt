#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : ltss_result
# @Date     : 05/30/2019 13:00:47
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


from __future__ import print_function

import os

import numpy as np


def result_node():
    path = '/network/rit/lab/ceashpc/share_data/BA/run6/ltss'
    fn = 'result_node.txt'

    results = {}
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            terms = line.strip().split(',')
            name = terms[0]
            func_type = terms[1]
            run_time = float(terms[3])
            prec = float(terms[4])
            rec = float(terms[5])
            fm = float(terms[6])

            if '1000_' in name and func_type == 'EMS':
                if 1000 not in results:
                    results[1000] = []
                else:
                    results[1000].append((run_time, prec, rec, fm))
            elif '2000' in name and func_type == 'EMS':
                if 2000 not in results:
                    results[2000] = []
                else:
                    results[2000].append((run_time, prec, rec, fm))
            elif '3000' in name and func_type == 'EMS':
                if 3000 not in results:
                    results[3000] = []
                else:
                    results[3000].append((run_time, prec, rec, fm))
            elif '4000' in name and func_type == 'EMS':
                if 4000 not in results:
                    results[4000] = []
                else:
                    results[4000].append((run_time, prec, rec, fm))
            elif '5000' in name and func_type == 'EMS':
                if 5000 not in results:
                    results[5000] = []
                else:
                    results[5000].append((run_time, prec, rec, fm))
            elif '6000' in name and func_type == 'EMS':
                if 6000 not in results:
                    results[6000] = []
                else:
                    results[6000].append((run_time, prec, rec, fm))
            elif '7000' in name and func_type == 'EMS':
                if 7000 not in results:
                    results[7000] = []
                else:
                    results[7000].append((run_time, prec, rec, fm))
            elif '8000' in name and func_type == 'EMS':
                if 8000 not in results:
                    results[8000] = []
                else:
                    results[8000].append((run_time, prec, rec, fm))
            elif '9000' in name and func_type == 'EMS':
                if 9000 not in results:
                    results[9000] = []
                else:
                    results[9000].append((run_time, prec, rec, fm))
            elif '10000' in name and func_type == 'EMS':
                if 10000 not in results:
                    results[10000] = []
                else:
                    results[10000].append((run_time, prec, rec, fm))

    for x in range(1, 11):
        result = results[x * 1000]
        print(x * 1000, np.mean(result, 0))


def result_edge():
    path = '/network/rit/lab/ceashpc/share_data/BA/run6/ltss'
    fn = 'result_edge.txt'

    results = {}
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            terms = line.strip().split(',')
            name = terms[0]
            func_type = terms[1]
            run_time = float(terms[3])
            prec = float(terms[4])
            rec = float(terms[5])
            fm = float(terms[6])

            if 'deg_3_' in name and func_type == 'EMS':
                if 3 not in results:
                    results[3] = []
                    results[3].append((run_time, prec, rec, fm))
                else:
                    results[3].append((run_time, prec, rec, fm))
            elif 'deg_4_' in name and func_type == 'EMS':
                if 4 not in results:
                    results[4] = []
                    results[4].append((run_time, prec, rec, fm))
                else:
                    results[4].append((run_time, prec, rec, fm))
            elif 'deg_5_' in name and func_type == 'EMS':
                if 5 not in results:
                    results[5] = []
                    results[5].append((run_time, prec, rec, fm))
                else:
                    results[5].append((run_time, prec, rec, fm))
            elif 'deg_6_' in name and func_type == 'EMS':
                if 6 not in results:
                    results[6] = []
                    results[6].append((run_time, prec, rec, fm))
                else:
                    results[6].append((run_time, prec, rec, fm))
            elif 'deg_7_' in name and func_type == 'EMS':
                if 7 not in results:
                    results[7] = []
                    results[7].append((run_time, prec, rec, fm))
                else:
                    results[7].append((run_time, prec, rec, fm))
            elif 'deg_8_' in name and func_type == 'EMS':
                if 8 not in results:
                    results[8] = []
                    results[8].append((run_time, prec, rec, fm))
                else:
                    results[8].append((run_time, prec, rec, fm))
            elif 'deg_9_' in name and func_type == 'EMS':
                if 9 not in results:
                    results[9] = []
                    results[9].append((run_time, prec, rec, fm))
                else:
                    results[9].append((run_time, prec, rec, fm))
            elif 'deg_10_' in name and func_type == 'EMS':
                if 10 not in results:
                    results[10] = []
                    results[10].append((run_time, prec, rec, fm))
                else:
                    results[10].append((run_time, prec, rec, fm))

    for x in range(3, 11):
        result = results[x]
        print(x, np.mean(result, 0))


def result_condmat():
    path = '/network/rit/lab/ceashpc/share_data/BA/run6/ltss'
    fn = 'result_condmat.txt'

    results = []
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            terms = line.strip().split(',')
            name = terms[0]
            func_type = terms[1]
            run_time = float(terms[3])
            prec = float(terms[4])
            rec = float(terms[5])
            fm = float(terms[6])

            if func_type == 'EMS':
                results.append((run_time, prec, rec, fm))

    results = np.array(results)
    print(results)
    print(np.mean(results, 0))


def result_wiki():
    path = '/network/rit/lab/ceashpc/share_data/BA/run6/ltss'
    fn = 'result_wiki.txt'

    results = []
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            terms = line.strip().split(',')
            name = terms[0]
            func_type = terms[1]
            run_time = float(terms[3])
            prec = float(terms[4])
            rec = float(terms[5])
            fm = float(terms[6])

            if func_type == 'EMS':
                results.append((run_time, prec, rec, fm))

    results = np.array(results)
    print(results)
    print(np.mean(results, 0))


def result_bj():
    path = '/network/rit/lab/ceashpc/share_data/BA/run6/ltss'
    fn = 'result_bj.txt'

    results = []
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            terms = line.strip().split(',')
            name = terms[0]
            func_type = terms[1]
            run_time = float(terms[3])
            prec = float(terms[4])
            rec = float(terms[5])
            fm = float(terms[6])

            if func_type == 'EMS':
                results.append((run_time, prec, rec, fm))

    results = np.array(results)
    print(results)
    print(np.mean(results, 0))


def result_dblp():
    path = '/network/rit/lab/ceashpc/share_data/DBLP/tmp_new/'
    # fn = 'ltss_m2std_dblp_3.txt'
    fn = 'eventTree_dblp_result_0.15.txt'

    results = []
    with open(os.path.join(path, fn)) as rfile:
        for line in rfile:
            terms = line.strip().split(',')
            name = terms[0]
            func_type = terms[1]
            run_time = float(terms[3])
            prec = float(terms[4])
            rec = float(terms[5])
            fm = float(terms[6])

            if func_type == 'EMS':
                print(line)
                results.append((run_time, prec, rec, fm))

    results = np.array(results)[:]
    print(results)
    print(np.mean(results, 0))


if __name__ == '__main__':

    # result_node()

    # result_edge()

    # result_condmat()

    # result_wiki()

    # result_bj()

    result_dblp()
    x = 1