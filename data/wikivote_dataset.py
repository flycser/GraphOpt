#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : wikivote_dataset
# @Date     : 07/17/2019 22:02:44
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


from __future__ import print_function

import os
import pickle


def convert():
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/app2/wikivote'
    train_dataset = []
    for id in range(10):
        fn = 'blocks_20_mu_5_true_1000_case_{}.pkl'.format(id)
        with open(os.path.join(path, fn), 'rb') as rfile:
            instance = pickle.load(rfile)

        train_dataset.append(instance)

    fn = 'train.pkl'
    with open(os.path.join(path, fn), 'wb') as wfile:
        pickle.dump(train_dataset, wfile)

def rename():

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/log/ghtp/block_wikivote'
    for fn in os.listdir(path):
        src = fn
        dst = os.path.splitext(fn)[0] + '_serial' + '.txt'
        os.rename(os.path.join(path, src), os.path.join(path, dst))
        # os.rename(src, dst)
        # print(dst)

if __name__ == '__main__':

    convert()


    # rename()