#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : bj_dataset
# @Date     : 07/17/2019 22:07:28
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


from __future__ import print_function

import os
import pickle


def convert():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/app2/bj'
    date = '20130913'
    train_dataset = []
    for hour in [17, 18]:
        for id in range(6):
            fn = '{}_hour_{}_ind_{}.pkl'.format(date, hour, id)
            with open(os.path.join(path, fn), 'rb') as rfile:
                instance = pickle.load(rfile)

            train_dataset.append(instance)
            # print(instance.keys())

    fn = 'train.pkl'
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(train_dataset, wfile)

def rename():

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/log/ghtp/block_bj'
    for fn in os.listdir(path):
        src = fn
        dst = os.path.splitext(fn)[0] + '_serial' + '.txt'
        os.rename(os.path.join(path, src), os.path.join(path, dst))
        # os.rename(src, dst)
        # print(dst)

if __name__ == '__main__':

    convert()