# run the experiments of dynamic graph-mp (DGMP)

import subprocess
import time
from multiprocessing import Pool, Lock
from sparse_learning.proj_algo import head_proj
from sparse_learning.proj_algo import tail_proj

import pickle
import EMS
import numpy as np
from post_process import *
import networkx as nx

logger = True
output_lock = Lock()


def print_log(log_file, string):
    if logger == True:
        # print(string)
        if log_file != None:
            outfile = open(log_file, "a")
            outfile.write(string)
            outfile.close()


def normalized_gradient(x, grad):
    # TODO what is the purpose of normalization
    # rescale gradient to a feasible space [0, 1]?
    normalized_grad = np.zeros_like(grad)
    count1 = 0
    count2 = 0
    for i in range(len(grad)):
        if grad[i] > 0.0 and x[i] == 1.0:
            normalized_grad[i] = 0.0
            count1 += 1
        elif grad[i] < 0.0 and x[i] == 0.0:
            normalized_grad[i] = 0.0
            count2 += 1
        else:
            normalized_grad[i] = grad[i]
    print("count1: {}, count2: {}".format(count1, count2))
    return normalized_grad



def dynamic_graph_mp(data, sparsity, max_iter, trade_off, top_k, log_file, func_name="EMS"):
    """
    :param func_name: score function name
    :param k: sparsity
    :param max_iter: max number of iterations
    :param G: networkx graph
    :param true_subgraph: a list of nodes that represents the ground truth subgraph
    :return: prediction xt, which denotes by the predicted abnormal nodes
    """
    features = data["features"]
    if func_name == "GlobalEMS":
        func = EMS.GlobalEMS(feature_matrix=features, trade_off=trade_off)
    else:
        print("ERROR")

    G = data["graph"]
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    costs = np.ones(num_edges)
    top_k_true_subgraphs = data["top_k_true_subgraphs"][:top_k]
    T = len(top_k_true_subgraphs[0])
    combined_true_subgraphs = [[] for i in range(T)]
    for true_subgraphs in top_k_true_subgraphs:
        for i in range(T):
            combined_true_subgraphs[i] += true_subgraphs[i]
    edges = np.array(G.edges())

    print_log(log_file, "\n----------------initialization---------------\n")
    X = func.get_init_point_random()
    XT = np.copy(X)
    #
    print_log(log_file, "\n------------------searching------------------\n")
    for iter in range(max_iter):
        Omega_X = []
        X_prev = np.copy(XT)
        print("iter: {}, time: {}".format(iter, time.asctime(time.localtime(time.time()))))
        for t in range(T):
            xt = XT[t]
            grad = func.get_loss_grad(XT, t)

            if 0 == iter:
                xt_zero = np.zeros_like(xt)
                normalized_grad = normalized_gradient(xt_zero, grad)  # rescale gradient of x into [0, 1]
            else:
                normalized_grad = normalized_gradient(xt, grad)  # rescale gradient of x into [0, 1]

            # g: number of connected component
            re_head = head_proj(edges=edges, weights=costs, x=normalized_grad, g=2*top_k, s=sparsity, budget=sparsity - 1, delta=1. / 169., max_iter=100,
                                err_tol=1e-8, root=-1, pruning='strong', epsilon=1e-10, verbose=0)
            re_nodes, re_edges, p_x = re_head
            gamma_xt = set(re_nodes)
            supp_xt = set([ind for ind, _ in enumerate(xt) if _ != 0.])

            omega_x = gamma_xt.union(supp_xt)
            if 0 == iter:
                omega_x = gamma_xt
            Omega_X.append(omega_x)

        BX = func.get_argmax_fx_with_proj(XT, Omega_X)  # TODO: how to solve this argmax correctly

        for t in range(T):
            bx = BX[t]
            re_tail = tail_proj(edges=edges, weights=costs, x=bx, g=2*top_k, s=sparsity, budget=sparsity - 1, nu=2.5, max_iter=100, err_tol=1e-8, root=-1,
                                pruning='strong', verbose=0)
            re_nodes, re_edges, p_x = re_tail
            psi_x = re_nodes
            xt = np.zeros_like(XT[t])
            xt[list(psi_x)] = bx[list(psi_x)]
            XT[t] = xt
        gap_x = np.linalg.norm(XT - X_prev) ** 2
        if gap_x < 1e-6:
            break

        print_log(log_file, '\ncurrent performance iteration: {}\n'.format(iter))
        obj_val, ems_score, smooth_penalty, binarized_penalty = func.get_obj_value(XT)
        print_log(log_file, 'trade-off: {}\n'.format(trade_off))
        print_log(log_file, 'objective value of prediction: {:5f}\n'.format(obj_val))
        print_log(log_file, 'global ems score of prediction: {:5f}\n'.format(ems_score))
        print_log(log_file, 'penalty of prediction: {:5f}\n'.format(obj_val - ems_score))

        pred_subgraphs = [np.nonzero(x)[0] for x in XT]
        print_log(log_file, "----------------- current predicted subgraphs:\n")
        for t in range(T):
            pred_sub = sorted(pred_subgraphs[t])
            print_log(log_file, "{}, {}\n".format(t, pred_sub))

        print_log(log_file, "---------------------------------------------:\n")
        for t in range(T):
            pred_sub = sorted(pred_subgraphs[t])
            x = np.round(XT[t][pred_sub], 5)
            fea = np.round(features[t][pred_sub], 5)
            print_log(log_file, "{}, {}\n".format(t, zip(pred_sub, x, fea)))

        print_log(log_file, "----------------- current true subgraphs:\n")
        for t in range(T):
            true_sub = sorted(combined_true_subgraphs[t])
            x = np.round(XT[t][true_sub], 5)
            fea = np.round(features[t][true_sub], 5)
            print_log(log_file, "{}, {}\n".format(t, zip(true_sub, x, fea)))

        global_prec, global_rec, global_fm, global_iou, valid_prec, valid_rec, valid_fm, valid_iou = evaluate(combined_true_subgraphs, pred_subgraphs)
        print_log(log_file, 'global_prec={:4f},\nglobal_rec={:.4f},\nglobal_fm={:.4f},\nglobal_iou={:.4f}\n'.format(global_prec, global_rec, global_fm, global_iou))
        print_log(log_file, 'valid_prec={:.4f},\nvalid_rec={:.4f},\nvalid_fm={:.4f},\nvalid_iou={:.4f}\n'.format(valid_prec, valid_rec, valid_fm, valid_iou))

    return XT


def worker(para):
    (data, data_type, case_id, func_name, max_iter, trade_off, sparsity, log_file, result_file, top_k) = para

    G = data["graph"]
    features = data["features"]
    # true_subgraphs = data["true_subgraphs"]
    top_k_true_subgraphs = data["top_k_true_subgraphs"][:top_k]
    T = len(top_k_true_subgraphs[0])
    combined_true_subgraphs = [[] for i in range(T)]
    for true_subgraphs in top_k_true_subgraphs:
        for i in range(T):
            combined_true_subgraphs[i] += true_subgraphs[i]
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    print_log(log_file, "---------------------------------Ground-Truth---------------------------------------\n")
    print_log(log_file, 'num of node: {}\n'.format(num_nodes))
    print_log(log_file, 'num of edges: {}\n'.format(num_edges))
    print_log(log_file, 'number of time stamps: {}\n'.format(T))
    print_log(log_file, 'all true subgraph: {}\n'.format(combined_true_subgraphs))

    true_X = []
    for true_sub in combined_true_subgraphs:
        true_x = np.zeros(num_nodes)
        true_x[true_sub] = 1.0
        true_X.append(true_x)

    if func_name == "GlobalEMS":
        func = EMS.GlobalEMS(feature_matrix=data["features"], trade_off=trade_off)
        obj_val, ems_score, smooth_penalty, binarized_penalty = func.get_obj_value(true_X)
    print_log(log_file, '\ntrade-off: {}\n'.format(trade_off))
    print_log(log_file, 'objective value of ground-truth: {:5f}\n'.format(obj_val))
    print_log(log_file, 'global eml score of ground-truth: {:5f}\n'.format(ems_score))
    print_log(log_file, 'penalty of ground-truth: {:5f}\n'.format(obj_val - ems_score))

    print_log(log_file, "\n-----------------------------Dynamic Graph-MP--------------------------------------")
    XT = dynamic_graph_mp(data, sparsity, max_iter, trade_off, top_k, log_file, func_name)

    print_log(log_file, "\n--------------------------Evaluation of Raw Prediction-------------------------------")
    raw_pred_subgraphs = [np.nonzero(x)[0] for x in XT]
    global_prec, global_rec, global_fm, global_iou, valid_prec, valid_rec, valid_fm, valid_iou = evaluate(combined_true_subgraphs, raw_pred_subgraphs, log_file)
    obj_val, ems_score, smooth_penalty, binarized_penalty = func.get_obj_value(XT)
    print_log(log_file, '\ntrade-off: {}\n'.format(trade_off))
    print_log(log_file, 'objective value of prediction: {:5f}\n'.format(obj_val))
    print_log(log_file, 'global ems score of prediction: {:5f}\n'.format(ems_score))
    print_log(log_file, 'smooth penalty of prediction: {:5f}\n'.format(obj_val - ems_score))
    # print('{}, {}, {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}\n'.format(
    #     trade_off, sparsity, case_id, global_prec, global_rec, global_fm, global_iou, valid_prec, valid_rec, valid_fm, valid_iou))

    print_log(log_file, "\n--------------------------Evaluation of Refined Prediction-------------------------------")
    edges, newid_oldid_dict = generate_combined_subgraphs(G, T, raw_pred_subgraphs)
    combined_G = nx.Graph()
    combined_G.add_edges_from(edges)
    # largest_cc = max(nx.connected_components(combined_G), key=len)
    # refined_pred_subgraphs = refine_predicted_subgraph(largest_cc, newid_oldid_dict, T)
    CC = [sorted(c) for c in sorted(nx.connected_components(combined_G), key=len, reverse=True) if len(c) > 5][:]
    all_pred_subgraphs = get_pred_subgraphs(CC, newid_oldid_dict, T)

    refined_pred_subgraphs = [[] for i in range(T)]
    for pred_subgraphs in all_pred_subgraphs:
        for i in range(T):
            refined_pred_subgraphs[i] += pred_subgraphs[i]

    refined_XT = []
    for t in range(T):
        x = np.zeros_like(XT[t])
        pred_sub = sorted(refined_pred_subgraphs[t])
        x[pred_sub] = XT[t][pred_sub]
        refined_XT.append(x)
    global_prec, global_rec, global_fm, global_iou, valid_prec, valid_rec, valid_fm, valid_iou = evaluate(combined_true_subgraphs, refined_pred_subgraphs, log_file)
    obj_val, ems_score, smooth_penalty, binarized_penalty = func.get_obj_value(refined_XT)
    print_log(log_file, '\nsmooth trade-off: {}\n'.format(trade_off))
    print_log(log_file, 'objective value of prediction: {:5f}\n'.format(obj_val))
    print_log(log_file, 'global ems score of prediction: {:5f}\n'.format(ems_score))
    print_log(log_file, 'temporal smooth penalty of prediction: {:5f}\n'.format(smooth_penalty))

    # print_log(log_file, "\n--------------------------Evaluation of Final Prediction-------------------------------")
    # final_pred_subgraphs, final_XT = deletion1(func, refined_XT, refined_pred_subgraphs)
    # global_prec, global_rec, global_fm, global_iou, valid_prec, valid_rec, valid_fm, valid_iou = evaluate(true_subgraphs, final_pred_subgraphs, log_file)
    # obj_val, ems_score, smooth_penalty, binarized_penalty = func.get_obj_value(final_XT)
    # print_log(log_file, '\nsmooth trade-off: {}\n'.format(trade_off))
    # print_log(log_file, 'objective value of prediction: {:5f}\n'.format(obj_val))
    # print_log(log_file, 'global ems score of prediction: {:5f}\n'.format(ems_score))
    # print_log(log_file, 'temporal smooth penalty of prediction: {:5f}\n'.format(smooth_penalty))

    # output_lock.acquire()
    # with open(result_file, "a") as f:
    #     f.write('{}, {}, {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}\n'.format(
    #         trade_off, sparsity, case_id, global_prec, global_rec, global_fm, global_iou, valid_prec, valid_rec, valid_fm, valid_iou))
    # output_lock.release()
    print('{}, {}, {}, {:.5f}, {:.5f}, {:.5f}\n'.format(trade_off, sparsity, case_id, global_prec, global_rec, global_fm))



def get_pred_subgraphs(CC, newid_oldid_dict, T):
    all_true_subgraphs = []
    for cc in CC:
        true_subgraphs = [[] for i in range(T)]
        for node in cc:
            t, old_id = newid_oldid_dict[node]
            true_subgraphs[t].append(old_id)
        print("total nodes: {}, pred_subgraphs {}".format(len(cc), true_subgraphs))
        all_true_subgraphs.append(true_subgraphs)
    return all_true_subgraphs




def single_test():
    graph_type = "Beijing"
    func_name = "GlobalEMS"
    max_iter = 10

    data_type = "test"
    date = 20130913
    hour = 17
    case_id = "{}_hour_{}".format(date, hour)
    data_dir = "{}/{}_hour_{}.pkl".format(graph_type,  date, hour)
    data = pickle.load(open(data_dir, "rb"))

    top_k = 30
    input_paras = []
    for trade_off in [x * 0.01 for x in range(1, 2)]:
        # for trade_off in [100]:
        for sparsity in [x*100 for x in range(10, 11)]:
            # log_file = "../logs/{}/run{}/{}/date_{}_hour_{}_tradeoff_{}_sparsity{}.out".format(
            #     graph_type, run, func_name, date, hour, trade_off, sparsity)
            # with open(log_file, "w") as f:
            #     f.write("trade_off= {}, sparsity={}\n".format(trade_off, sparsity))
            log_file = None
            # log_file = "logs/{}/run{}/{}/nodes_{}_windows_{}_mu_{}_subsize_{}_{}_range_{}_{}_overlap_{}_{}_{}_{}.out".format(
            #     graph_type, run, func_name, num_nodes, num_time_slots, mu1, subsize_min, subsize_max,
            #     start_time, end_time, overlap, pattern, data_type, case_id)
            # with open(log_file, "w") as f:
            #     f.write("trade_off= {}, sparsity={}\n".format(trade_off, sparsity))

            result_file = None
            # result_file = "../results/{}/run{}/{}/all_results.out".format(graph_type, run, func_name)
            # with open(result_file, "w") as f:
            #     f.write("trade_off, sparsity, case_id, global_prec, global_rec, global_fm, global_iou, true_prec, true_rec, true_fm, true_iou\n")

            para = (data, data_type, case_id, func_name, max_iter, trade_off, sparsity, log_file, result_file, top_k)
            input_paras.append(para)
            worker(para)
    # pool = Pool(processes=50)
    # pool.map(worker, input_paras)
    # pool.close()
    # pool.join()


if __name__ == "__main__":
    single_test()
