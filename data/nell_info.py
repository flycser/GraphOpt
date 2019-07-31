#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : nell_info
# @Date     : 05/30/2019 22:07:17
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


from __future__ import print_function

import os
import cPickle

import numpy as np
import  networkx as nx


if __name__ == '__main__':

    path = '/network/rit/lab/ceashpc/fjie/tmp/nell_data'
    fn = 'ind.nell.0.001.ally'

    with open(os.path.join(path, fn)) as rfile:
        features = cPickle.load(rfile)

    print(features.shape)
    print(features[0])

