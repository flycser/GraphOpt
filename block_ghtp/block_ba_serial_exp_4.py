#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : block_ba_serial_exp_4
# @Date     : 07/28/2019 14:44:50
# @Poject   : GraphOpt
# @Author   : FEI, hfut_jf@aliyun.com
# @Desc     :


from __future__ import print_function

import sys
import pickle
from multiprocessing import Pool, Lock
import time
import networkx as nx
import numpy as np
from sparse_learning.proj_algo import head_proj
from sparse_learning.proj_algo import tail_proj

import BlockSumEMS

# logger = True
logger = False
output_lock = Lock()


def print_log(log_file, string):
    if logger == True:
        # print(string)
        if log_file != None:
            outfile = open(log_file, "a")
            outfile.write(string)
            outfile.close()


def find_best_trade_off(result_file):
    """find the best trade-off based on the best performance on global-iou of raw prediction"""
    all_global_prec, all_global_rec, all_global_fm, all_global_iou = {}, {}, {}, {}
    with open(result_file, "r") as f:
        f.readline()
        for line in f:
            fields = line.strip().split(',')
            trade_off = float(fields[0])
            sparsity = int(fields[1])
            case_id = int(fields[2])
            global_prec = float(fields[3])
            global_rec = float(fields[4])
            global_fm = float(fields[5])
            global_iou = float(fields[6])

            key = (trade_off, sparsity)
            if key not in all_global_prec:
                all_global_prec[key] = []
            if key not in all_global_rec:
                all_global_rec[key] = []
            if key not in all_global_fm:
                all_global_fm[key] = []
            if key not in all_global_iou:
                all_global_iou[key] = []

            all_global_prec[key].append(global_prec)
            all_global_rec[key].append(global_rec)
            all_global_fm[key].append(global_fm)
            all_global_iou[key].append(global_iou)

    (best_trade_off, best_sparsity) = sorted(all_global_iou, key=lambda x: np.mean(all_global_iou[x]), reverse=True)[0]
    # sorted_dict = sorted(all_global_iou.iteritems(), key= lambda (x, y):np.mean(y), reverse=True)
    # for key, value in sorted_dict:
    #     print key, np.mean(value), value
    avg_global_prec = np.mean(all_global_prec[(best_trade_off, best_sparsity)])
    avg_global_rec = np.mean(all_global_rec[(best_trade_off, best_sparsity)])
    avg_global_fm = np.mean(all_global_fm[(best_trade_off, best_sparsity)])
    avg_global_iou = np.mean(all_global_iou[(best_trade_off, best_sparsity)])


    print("{}\n{}\n{}\n{}\n{}\n{}\n".format(best_trade_off, best_sparsity, avg_global_prec,
                                            avg_global_rec, avg_global_fm, avg_global_iou))
    return best_trade_off, best_sparsity




def normalized_gradient(x, grad):
    # TODO what is the purpose of normalization
    # rescale gradient to a feasible space [0, 1]?
    normalized_grad = np.zeros_like(grad)
    for i in range(len(grad)):
        if grad[i] > 0.0 and x[i] == 1.0:
            normalized_grad[i] = 0.0
        elif grad[i] < 0.0 and x[i] == 0.0:
            normalized_grad[i] = 0.0
        else:
            normalized_grad[i] = grad[i]
    return normalized_grad


def relabel_nodes(nodes_set):
    """key is global node id, value is id in the block, which starts from 0"""
    nodes_id_dict = {}
    for nodes in nodes_set:
        ind = 0
        for n in sorted(nodes):
            nodes_id_dict[n] = ind
            ind += 1
    return nodes_id_dict


def relabel_edges(G, nodes_set, nodes_id_dict):
    relabeled_edges_set = []
    for nodes in nodes_set:
        edges = []
        for (u, v) in G.subgraph(nodes).edges():
            edges.append((nodes_id_dict[u], nodes_id_dict[v]))
        relabeled_edges_set.append(edges)
    return np.array(relabeled_edges_set)


def get_boundary_xs(X, boundary_edges, nodes_id_dict):
    """get the boundary_xs_dict, which key is boundary edge with node_id in block relabeled,
    value is the x-val of adj node """
    boundary_xs_dict = {}
    for (u, v) in boundary_edges:
        node_id_in_block = nodes_id_dict[u]
        adj_x_val = X[v]
        boundary_xs_dict[(node_id_in_block, v)] = adj_x_val
    return boundary_xs_dict


def evaluate(true_subgraph, pred_subgraph, log_file=None):
    true_subgraph, pred_subgraph = set(list(true_subgraph)), set(list(pred_subgraph))
    if log_file != None:
        print_log(log_file, "\nsize: {}, pred subgraph: {}\n".format(len(pred_subgraph), sorted(pred_subgraph)))
        print_log(log_file, "size: {}, true subgraph: {}\n".format(len(true_subgraph), sorted(true_subgraph)))
    intersection = true_subgraph.intersection(pred_subgraph)
    union = true_subgraph.union(pred_subgraph)

    if len(pred_subgraph) != 0.:
        global_prec = len(intersection) / float(len(pred_subgraph))
    else:
        global_prec = 0.
    global_rec = len(intersection) / float(len(true_subgraph))
    if global_prec + global_rec != 0.:
        global_fm = (2. * global_prec * global_rec) / (global_prec + global_rec)
    else:
        global_fm = 0.
    global_iou = len(intersection) / float(len(union))

    if log_file != None:
        print_log(log_file, "\n----------------------Overall Performance------------------\n")
        print_log(log_file, "global_prec:{},\nglobal_rec:{},\nglobal_fm:{},\nglobal_iou:{}".format(global_prec, global_rec, global_fm, global_iou))
    return global_prec, global_rec, global_fm, global_iou


def block_graph_mp(data, k, max_iter, trade_off, log_file, func_name="EMS"):
    """
    :param func_name: score function name
    :param k: sparsity
    :param max_iter: max number of iterations
    :param G: networkx graph
    :param true_subgraph: a list of nodes that represents the ground truth subgraph
    :return: prediction xt, which denotes by the predicted abnormal nodes
    """
    G = data["graph"]
    features = np.array(data["features"])
    nodes_set = data["nodes_set"]
    boundary_edges_dict = data["block_boundary_edges_dict"]
    node_block_dict = data["node_block_dict"]
    true_subgraph = sorted(data["true_subgraph"])
    num_blocks = len(boundary_edges_dict)
    nodes_id_dict = relabel_nodes(nodes_set)  # key is global node id, value is local node id
    if func_name == "BlockSumEMS":
        func = BlockSumEMS.BlockSumEMS(features=features, num_blocks=num_blocks, nodes_set=nodes_set,
                                         boundary_edges_dict=boundary_edges_dict,
                                         nodes_id_dict=nodes_id_dict, trade_off=trade_off)
    else:
        print("ERROR")

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    nodes_id_dict = relabel_nodes(nodes_set)  # key is global node id, value is local node id
    relabeled_edges_set = relabel_edges(G, nodes_set, nodes_id_dict)

    print_log(log_file, "\n----------------initialization---------------\n")
    X = func.get_init_point_random()
    XT = np.copy(X)
    print_log(log_file, "\n------------------searching------------------\n")
    start_time = time.time()
    learning_rate = 1.
    for iter in range(max_iter):
        Omega_X = []
        X_prev = np.copy(XT)
        print("iter: {}, time: {}".format(iter, time.asctime(time.localtime(time.time()))))
        for t in range(num_blocks):
            xt = XT[sorted(nodes_set[t])]
            boundary_xs_dict = get_boundary_xs(XT, boundary_edges_dict[t], nodes_id_dict)  # key is boundary edge, value is adjacent x in other blocks
            fea = features[sorted(nodes_set[t])]
            grad = func.get_loss_grad(xt, fea, boundary_xs_dict)

            if 0 == iter:  # because we initialize the x as 0.000001 to avoid the divided by zero error when calculating the gradient
                xt_zero = np.zeros_like(xt)
                normalized_grad = normalized_gradient(xt_zero, grad)
            else:
                normalized_grad = normalized_gradient(xt, grad)

            # g: number of connected component
            edges = np.array(relabeled_edges_set[t])
            costs = np.ones(len(edges))
            re_head = head_proj(edges=edges, weights=costs, x=normalized_grad, g=1, s=k, budget=k - 1, delta=1. / 169., max_iter=100, err_tol=1e-8, root=-1, pruning='strong', epsilon=1e-10, verbose=0)
            re_nodes, re_edges, p_x = re_head
            gamma_xt = set(re_nodes)
            supp_xt = set([ind for ind, _ in enumerate(xt) if _ != 0.])

            indicator_x = np.zeros(len(xt))
            indicator_x[list(gamma_xt)] = 1.
            if iter == 0:
                tmp_x = np.zeros_like(xt) + learning_rate * grad * indicator_x  # note, not update current variables, only use the intermediate results
            else:
                tmp_x = xt + learning_rate * grad * indicator_x
            omega_x = set([ind for ind, _ in enumerate(tmp_x) if not 0. == _])

            # if 0 == iter:  # because we initialize the x as 0.000001 to avoid the divided by zero error when calculating the gradient
            #     omega_x = gamma_xt
            Omega_X.append(omega_x)

        print("---Head Projection Finished: time: {}".format(time.asctime(time.localtime(time.time()))))
        # BX = func.get_argmax_fx_with_proj_accelerated(XT, Omega_X, iter)  # TODO: how to solve this argmax correctly
        BX = func.get_argmax_fx_with_proj_accelerated_2(XT, Omega_X)  # TODO: how to solve this argmax correctly

        print("---ArgMax Finished: time: {}".format(time.asctime(time.localtime(time.time()))))
        for t in range(num_blocks):
            edges = np.array(relabeled_edges_set[t])
            costs = np.ones(len(edges))
            bx = BX[nodes_set[t]]
            re_tail = tail_proj(edges=edges, weights=costs, x=bx, g=1, s=k, budget=k - 1, nu=2.5, max_iter=100, err_tol=1e-8, root=-1,
                                pruning='strong', verbose=0)
            re_nodes, re_edges, p_x = re_tail
            psi_x = re_nodes
            xt = np.zeros_like(XT[nodes_set[t]])
            xt[list(psi_x)] = bx[list(psi_x)]  # TODO: note the non-zero entries of xt[list(psi_x)] may not be connected
            XT[nodes_set[t]] = xt

        print("---Tail Projection Finished: time: {}".format(time.asctime(time.localtime(time.time()))))
        gap_x = np.linalg.norm(XT - X_prev) ** 2
        if gap_x < 1e-6:
            break

        print_log(log_file, '\ncurrent performance iteration: {}\n'.format(iter))
        obj_val, ems_score, smooth_penalty, binarized_penalty = func.get_obj_value(XT, boundary_edges_dict)
        print_log(log_file, 'trade-off: {}\n'.format(trade_off))
        print_log(log_file, 'objective value of prediction: {:5f}\n'.format(obj_val))
        print_log(log_file, 'global ems score of prediction: {:5f}\n'.format(ems_score))
        print_log(log_file, 'penalty of prediction: {:5f}\n'.format(obj_val - ems_score))

        pred_subgraph = sorted(np.nonzero(XT)[0])
        print_log(log_file, "----------------- current predicted subgraph vs true subgraph:\n")
        print_log(log_file, "{}, size: {}\n".format(pred_subgraph, len(pred_subgraph)))
        print_log(log_file, "{}, size: {}\n".format(true_subgraph, len(true_subgraph)))

        print_log(log_file, "----------------- info of current predicted subgraph:\n")
        fea = np.round(features[pred_subgraph], 5)
        print_log(log_file, "{}\n".format(zip(pred_subgraph, np.round(XT[pred_subgraph], 5), fea)))

        print_log(log_file, "----------------- info of current true subgraph:\n")
        fea = np.round(features[true_subgraph], 5)
        print_log(log_file, "{}\n".format(zip(true_subgraph, np.round(XT[true_subgraph], 5), fea)))

        global_prec, global_rec, global_fm, global_iou = evaluate(true_subgraph, pred_subgraph)
        print_log(log_file, 'global_prec={:4f},\nglobal_rec={:.4f},\nglobal_fm={:.4f},\nglobal_iou={:.4f}\n'.format(global_prec, global_rec, global_fm, global_iou))

    end_time = time.time()
    total_time = end_time - start_time
    print("time {}".format(total_time))

    return XT, total_time


def worker(para):
    (data, data_type, case_id, func_name, max_iter, trade_off, sparsity, log_file, result_file) = para

    G = data["graph"]
    features = np.array(data["features"])
    true_subgraph = data["true_subgraph"]
    boundary_edges_dict = data["block_boundary_edges_dict"]
    nodes_set = data["nodes_set"]
    num_blocks = len(boundary_edges_dict)
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    print_log(log_file, "---------------------------------Ground-Truth---------------------------------------\n")
    print_log(log_file, 'num of node: {}\n'.format(num_nodes))
    print_log(log_file, 'num of edges: {}\n'.format(num_edges))
    print_log(log_file, 'number of blocks: {}\n'.format(num_blocks))
    print_log(log_file, 'all true subgraph: {}\n'.format(true_subgraph))
    print_log(log_file, 'true subgraph size: {}\n'.format(len(true_subgraph)))
    true_X = np.zeros(num_nodes)
    true_X[true_subgraph] = 1.0
    func = BlockSumEMS.BlockSumEMS(features=features, num_blocks=num_blocks, trade_off=trade_off, nodes_set=nodes_set)
    obj_val, ems_score, smooth_penalty, binarized_penalty = func.get_obj_value(true_X, boundary_edges_dict)
    print_log(log_file, '\ntrade-off: {}\n'.format(trade_off))
    print_log(log_file, 'objective value of ground-truth: {:5f}\n'.format(obj_val))
    print_log(log_file, 'global eml score of ground-truth: {:5f}\n'.format(ems_score))
    print_log(log_file, 'penalty of ground-truth: {:5f}\n'.format(obj_val - ems_score))

    print_log(log_file, "\n-----------------------------Block Graph-MP--------------------------------------")
    X, total_time = block_graph_mp(data, sparsity, max_iter, trade_off, log_file, func_name)

    print_log(log_file, "\n-------------------------Evaluation of Raw Prediction----------------------------")
    raw_pred_subgraph = sorted(np.nonzero(X)[0])
    raw_global_prec, raw_global_rec, raw_global_fm, raw_global_iou = evaluate(true_subgraph, raw_pred_subgraph, log_file)
    obj_val, ems_score, smooth_penalty, binarized_penalty = func.get_obj_value(X, boundary_edges_dict)
    print_log(log_file, '\ntrade-off: {}\n'.format(trade_off))
    print_log(log_file, 'objective value of prediction: {:5f}\n'.format(obj_val))
    print_log(log_file, 'global ems score of prediction: {:5f}\n'.format(ems_score))
    print_log(log_file, 'smooth penalty of prediction: {:5f}\n'.format(obj_val - ems_score))
    print('\nRaw Prediction    : {}, {}, {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}'.format(
        trade_off, sparsity, case_id, raw_global_prec, raw_global_rec, raw_global_fm, raw_global_iou))

    print_log(log_file, "\n-----------------------Evaluation of Refined Prediction--------------------------")
    S = G.subgraph(raw_pred_subgraph)
    largest_cc = max(nx.connected_components(S), key=len)
    refined_pred_subgraph = sorted([node for node in largest_cc])
    refined_X = np.zeros_like(X)
    refined_X[refined_pred_subgraph] = X[refined_pred_subgraph]
    refined_global_prec, refined_global_rec, refined_global_fm, refined_global_iou = evaluate(true_subgraph, refined_pred_subgraph, log_file)
    obj_val, ems_score, smooth_penalty, binarized_penalty = func.get_obj_value(refined_X, boundary_edges_dict)
    print_log(log_file, '\nsmooth trade-off: {}\n'.format(trade_off))
    print_log(log_file, 'objective value of prediction: {:5f}\n'.format(obj_val))
    print_log(log_file, 'global ems score of prediction: {:5f}\n'.format(ems_score))
    print_log(log_file, 'smooth penalty of prediction: {:5f}\n'.format(smooth_penalty))
    print('Refined Prediction: {}, {}, {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}\n'.format(
        trade_off, sparsity, case_id, refined_global_prec, refined_global_rec, refined_global_fm, refined_global_iou))

    output_lock.acquire()
    with open(result_file, "a") as f:
        f.write('{}, {}, {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}\n'.format(
            trade_off, sparsity, case_id, total_time, raw_global_prec, raw_global_rec, raw_global_fm, raw_global_iou, refined_global_prec, refined_global_rec, refined_global_fm, refined_global_iou))
    output_lock.release()


def multiworkers(graph_type, run, num_nodes, num_blocks, mu1, true_size_min, true_size_max, data_type, func_name, max_iter,
                 best_trade_off=None, best_sparsity=None):
    if data_type == "train":
        data_dir = "/network/rit/lab/ceashpc/share_data/{}/run{}/nodes_{}_blocks_{}_mu_{}_subsize_{}_{}_{}.pkl".format(
            graph_type, run, num_nodes, num_blocks, mu1, true_size_min, true_size_max, data_type)
        all_data = pickle.load(open(data_dir, "rb"))
        input_paras = []
        for trade_off in [x * 0.01 for x in range(80, 121)]:
            for sparsity in [50]:
                for case_id in range(100):
                    data = all_data[case_id]
                    # log_file = "../logs/{}/run{}/{}/nodes_{}_blocks_{}_mu_{}_subsize_{}_{}_{}_{}_tradeoff_{}_sparsity_{}.out".format(
                    #     graph_type, run, func_name, num_nodes, num_blocks, mu1, true_size_min, true_size_max, data_type, case_id,
                    #     trade_off, sparsity)
                    # with open(log_file, "w") as f:
                    #     f.write("trade_off= {}, sparsity={}\n".format(trade_off, sparsity))
                    log_file = None
                    result_file = "../results/{}/run{}/{}/nodes_{}_blocks_{}_mu_{}_subsize_{}_{}_{}.out".format(
                        graph_type, run, func_name, num_nodes, num_blocks, mu1, true_size_min, true_size_max, data_type)
                    para = (data, data_type, case_id, func_name, max_iter, trade_off, sparsity, log_file, result_file)
                    input_paras.append(para)
        pool = Pool(processes=50)
        pool.map(worker, input_paras)
        pool.close()
        pool.join()

    elif data_type == "test":
        data_dir = "/network/rit/lab/ceashpc/share_data/{}/run{}/nodes_{}_blocks_{}_mu_{}_subsize_{}_{}_{}.pkl".format(
            graph_type, run, num_nodes, num_blocks, mu1, true_size_min, true_size_max, data_type)
        all_data = pickle.load(open(data_dir, "rb"))
        input_paras = []
        for test_id in range(100):
            data = all_data[test_id]
            log_file = "../logs/{}/run{}/{}/nodes_{}_blocks_{}_mu_{}_subsize_{}_{}_{}_{}_tradeoff_{}_sparsity_{}.out".format(
                graph_type, run, func_name, num_nodes, num_blocks, mu1, true_size_min, true_size_max, data_type, test_id,
                best_trade_off, best_sparsity)
            with open(log_file, "w") as f:
                f.write("trade_off= {}, sparsity={}\n".format(best_trade_off, best_sparsity))
            result_file = "../results/{}/run{}/{}/nodes_{}_blocks_{}_mu_{}_subsize_{}_{}_{}.out".format(
                graph_type, run, func_name, num_nodes, num_blocks, mu1, true_size_min, true_size_max, data_type)
            para = (data, data_type, test_id, func_name, max_iter, best_trade_off, best_sparsity, log_file, result_file)
            input_paras.append(para)
        pool = Pool(processes=50)
        pool.map(worker, input_paras)
        pool.close()
        pool.join()


def main():
    graph_type = "Grid"
    run = 9
    num_nodes = 2500
    num_blocks = 25
    func_name = "BlockSumEMS"
    max_iter = 10
    data_type = "train"

    for mu1 in [3]:
        for true_size_min, true_size_max in [(100, 500)]:
            # multiworkers(graph_type, run, num_nodes, num_blocks, mu1, true_size_min=true_size_min, true_size_max=true_size_max,
            #              data_type="train", func_name=func_name, max_iter=max_iter)
            #
            result_file = "../results/{}/run{}/{}/nodes_{}_blocks_{}_mu_{}_subsize_{}_{}_{}.out".format(
                graph_type, run, func_name, num_nodes, num_blocks, mu1, true_size_min, true_size_max, data_type)
            print("mu = {}".format(mu1))
            best_trade_off, best_sparsity = find_best_trade_off(result_file)

            # data_type = "test"
            # multiworkers(graph_type, run, num_nodes, T, mu1, subsize_min, subsize_max, start_time, end_time, overlap, pattern,
            #              data_type="test", func_name=func_name, max_iter=max_iter,
            #              best_trade_off=best_trade_off, best_sparsity=best_sparsity)


def single_test():
    graph_type = "BA"
    run = 1
    num_nodes = 100000
    num_blocks = 100
    func_name = "BlockSumEMS"
    max_iter = 10
    true_size_min = 10000
    true_size_max = 10000
    data_type = "train"
    mu1 = 6

    case_id = 0
    data_dir = "/network/rit/lab/ceashpc/share_data/{}/run{}/nodes_{}_blocks_{}_mu_{}_subsize_{}_{}_{}.pkl".format(
        graph_type, run, num_nodes, num_blocks, mu1, true_size_min, true_size_max, data_type)
    all_data = pickle.load(open(data_dir, "rb"))
    data = all_data[case_id]
    true_subgraph = data["true_subgraph"]
    print("true_subgraph", true_subgraph)
    input_paras = []
    for trade_off in [x * 0.01 for x in range(1, 2)]:
        for sparsity in range(500, 501):
            log_file = "../logs/{}/run{}/{}/nodes_{}_blocks_{}_mu_{}_subsize_{}_{}_{}_{}_tradeoff_{}_sparsity_{}.out".format(
                graph_type, run, func_name, num_nodes, num_blocks, mu1, true_size_min, true_size_max, data_type, case_id,
                trade_off, sparsity)
            with open(log_file, "w") as f:
                f.write("trade_off= {}, sparsity={}\n".format(trade_off, sparsity))
            # log_file = None
            result_file = "../results/{}/run{}/{}/nodes_{}_blocks_{}_mu_{}_subsize_{}_{}_{}.out".format(
                graph_type, run, func_name, num_nodes, num_blocks, mu1, true_size_min, true_size_max, data_type)
            para = (data, data_type, case_id, func_name, max_iter, trade_off, sparsity, log_file, result_file)
            input_paras.append(para)
            worker(para)
    # pool = Pool(processes=50)
    # pool.map(worker, input_paras)
    # pool.close()
    # pool.join()


def run_edge():
    num_nodes = 100000
    num_blocks = 100
    func_name = "BlockSumEMS"
    max_iter = 10
    data_type = "train"
    deg = int(sys.argv[1])
    trade_off = 0.001

    sparsity = int(num_nodes / num_blocks / 2.0)
    data_dir = '/network/rit/lab/ceashpc/share_data/GraphOpt/app2/ba/train_{}_deg_{}.pkl'.format(num_nodes, deg)
    with open(data_dir, 'rb') as rfile:
        dataset = pickle.load(rfile)

    for case_id in range(len(dataset)):
        print(case_id)
        data = dataset[case_id]
        log_file = None
        result_file = "/network/rit/lab/ceashpc/share_data/GraphOpt/log/ghtp/block_ba_5/node_{}_{}_{}_{}_serial.out".format(num_nodes, deg, sparsity, trade_off)
        paras = (data, data_type, case_id, func_name, max_iter, trade_off, sparsity, log_file, result_file)
        worker(paras)


if __name__ == '__main__':

    run_edge()
