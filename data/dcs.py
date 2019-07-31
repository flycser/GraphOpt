#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : dcs
# @Date     : 07/18/2019 17:23:43
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     : some preprocess about dcs application


from __future__ import print_function

import os
import pickle

import numpy as np

def select_top_nodes(id):
    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions/syn'
    fn = 'attributes_biased_{}.pkl'.format(id)
    with open(os.path.join(path, fn), 'rb') as rfile:
        dataset = pickle.load(rfile)

        attributes = dataset['attributes']
        indices = np.argsort(attributes)[::-1][:10]
        print(indices)
        print(attributes[indices])

    path = '/network/rit/lab/ceashpc/share_data/GraphOpt/datasets/epinions/exp_1/case_{}'.format(id)
    for i, v in enumerate(indices):
        print(i, v)
        fn = '04Seeds_{}.txt'.format(i)
        with open(os.path.join(path, fn), 'w') as wfile:
            wfile.write('{}'.format(v))

        wpath = os.path.join(path, 'Output_{}'.format(i))
        os.mkdir(wpath)

if __name__ == '__main__':
    id = 0
    select_top_nodes(id)