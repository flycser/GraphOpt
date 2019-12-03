# Graph-Structured Optimization Algorithms

## Graph-Structured Matching Pursuit (Graph-MP)



## Graph-Structured Iterative Hard Thresholding (Graph-IHT)

## Graph-Structured Gradient Hard Thresholding Pursuit (Graph-GHTP)

## Subspace Graph-Structured Matching Pursuit

## Graph Block-Structured Matching Pursuit (GB-MP)

## Graph Block-Structured Iterative Hard Thresholding (GB-IHT)

### evo

1. synthetic, train
2. dc, train
3. bwsn, test
4. bj, train

## Graph Block-Structured Gradient Hard Thresholding Pursuit (GB-GHTP)

### evo

1. synthetic, test
2. dc, test
3. bwsn, complete
4. bj, train



## parameters

GB-IHT, step size [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1], trade off [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1], sparsity

GB-GHTP, step size [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1], trade off [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1], sparsity, argmin step size, argmin max iteration

## Note

BWSN dataset does not take **average** on multiple randomly generated samples. !!!

DC Beijing use another period data

source /network/rit/lab/ceashpc/fjie/venv/py2/bin/activate
cd ~/projects/GraphOpt/block_ghtp/
python evo_dc_exp.py > /network/rit/lab/ceashpc/share_data/GraphOpt/log/ghtp/sparsity=100.txt


## Datasets

### Anomalous Evolving Subgraph Detection

**Synthetic**

**BWSN**

**Washingtong D.C.**

**Beijing**

###



[comment]: <> (Chen, Feng, and Baojian Zhou. "A Generalized Matching Pursuit Approach for Graph-Structured Sparsity." IJCAI. 2016.)


[comment]: <> (Zhou, Baojian, and Feng Chen. "Graph-structured sparse optimization for connected subgraph detection." 2016 IEEE 16th International Conference on Data Mining (ICDM). IEEE, 2016. )

[comment]: <> (Chen, Feng, et al. "A generic framework for interesting subspace cluster detection in multi-attributed networks." 2017 IEEE International Conference on Data Mining (ICDM). IEEE, 2017.)
