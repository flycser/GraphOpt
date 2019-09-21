#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : extract_results_2
# @Date     : 08/11/2019 18:03:00
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


from __future__ import print_function

import numpy as np

def extract():
    fn = '/network/rit/lab/ceashpc/fjie/projects/GraphOpt/references/dcs/slurm-239751.out'
    with open(fn) as rfile:
        for line in rfile:
            if line.startswith('lambda'):
                terms = line.strip().split()
                print(terms[1], end='\t')

            if line.startswith('mean'):
                terms = line[line.find('[')+1:line.find(']')].split()
                for term in terms:
                    print(term, end='\t')

            if line.startswith('std'):
                terms = line[line.find('[')+1:line.find(']')].split()
                for term in terms:
                    print(term, end='\t')

                print()




if __name__ == '__main__':
    pass

    extract()