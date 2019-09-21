#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : extract_converge
# @Date     : 08/13/2019 22:01:44
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :

from __future__ import print_function


if __name__ == '__main__':
    fn = '/network/rit/lab/ceashpc/fjie/projects/GraphOpt/block_iht/slurm-240665.out'
    # fn = '/network/rit/lab/ceashpc/fjie/projects/GraphOpt/block_ghtp/slurm-240730.out'
    x = []
    with open(fn) as rfile:
        while True:
            line = rfile.readline().strip()
            if line.startswith('instance: 2'):
                count = 0
                while True:
                    line = rfile.readline().strip()
                    if line.startswith('objective value'):
                        obj_val = float(line.split(',')[0].split(':')[1])
                        print(count, obj_val)
                        x.append(obj_val)
                        count += 1

                    if count == 30:
                        break

                break

    for i in x:
        print(i, end=',')