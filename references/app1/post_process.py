import numpy as np


def generate_combined_subgraphs(G, T, pred_subgraphs):
    """output the predicted subgraphs of multiple time-stamps as one graph, and give new ID to same nodes at different time stamp"""
    num_nodes = len(G.nodes())
    pred_sub = pred_subgraphs[0]
    S = G.subgraph(pred_sub)
    newid_oldid_dict = {}
    oldid_newid_dict = {0: {}}  # first key is the time-stamp, and the second key is the oldid
    new_id = -1
    edges = []
    feature_vec = []
    weights = []

    for node in S.nodes():
        if node not in oldid_newid_dict[0]:
            new_id += 1
            oldid_newid_dict[0][node] = new_id
            newid_oldid_dict[new_id] = (0, node)
    for (u, v) in S.edges():
        edges.append((oldid_newid_dict[0][u], oldid_newid_dict[0][v]))

    for t in range(1, T):
        if t not in oldid_newid_dict:
            oldid_newid_dict[t] = {}
        pred_sub = pred_subgraphs[t]
        overlap = set(pred_subgraphs[t]).intersection(pred_subgraphs[t - 1])
        S = G.subgraph(pred_sub)

        for node in S.nodes():
            if node not in oldid_newid_dict[t]:
                new_id += 1
                oldid_newid_dict[t][node] = new_id
                newid_oldid_dict[new_id] = (t, node)
        for (u, v) in S.edges():
            edges.append((oldid_newid_dict[t][u], oldid_newid_dict[t][v]))

        for node_id in overlap:
            current_id = oldid_newid_dict[t][node_id]
            prev_id = oldid_newid_dict[t - 1][node_id]
            edges.append((current_id, prev_id))
    # return np.array(edges), np.array(feature_vec), np.array(weights), newid_oldid_dict
    return np.array(edges), newid_oldid_dict


def refine_predicted_subgraph(largest_cc, newid_oldid_dict, T):
    refined_pred_subgraphs = [[] for i in range(T)]

    for node in largest_cc:
        t, old_id = newid_oldid_dict[node]
        refined_pred_subgraphs[t].append(old_id)
    print("refined_pred_subgraphs {}".format(refined_pred_subgraphs))
    return refined_pred_subgraphs


def evaluate(true_subgraphs, pred_subgraphs, log_file=None):
    T = len(true_subgraphs)
    true_subgraphs_size = 0.
    pred_subgraphs_size = 0.
    valid_pred_subgraphs_size = 0.
    valid_intersection = 0.
    valid_union = 0.
    all_intersection = 0.
    all_union = 0.
    for t in range(T):
        true_subgraph, pred_subgraph = set(list(true_subgraphs[t])), set(list(pred_subgraphs[t]))
        true_subgraphs_size += len(true_subgraph)
        pred_subgraphs_size += len(pred_subgraph)
        intersection = true_subgraph.intersection(pred_subgraph)
        union = true_subgraph.union(pred_subgraph)
        all_intersection += len(intersection)
        all_union += len(union)

        if len(true_subgraph) != 0:
            valid_pred_subgraphs_size += len(pred_subgraph)
            valid_intersection += len(intersection)
            valid_union += len(union)

    if pred_subgraphs_size != 0.:
        global_prec = all_intersection / float(pred_subgraphs_size)
    else:
        global_prec = 0.
    global_rec = all_intersection / float(true_subgraphs_size)
    if global_prec + global_rec != 0.:
        global_fm = (2. * global_prec * global_rec) / (global_prec + global_rec)
    else:
        global_fm = 0.
    global_iou = all_intersection / float(all_union)

    if valid_pred_subgraphs_size != 0.:
        valid_global_prec = valid_intersection / float(valid_pred_subgraphs_size)
    else:
        valid_global_prec = 0.
    valid_global_rec = valid_intersection / true_subgraphs_size
    if valid_global_prec + valid_global_rec != 0.:
        valid_global_fm = (2. * valid_global_prec * valid_global_rec) / (valid_global_prec + valid_global_rec)
    else:
        valid_global_fm = 0.
    valid_global_iou = valid_intersection / float(valid_union)

    return global_prec, global_rec, global_fm, global_iou, valid_global_prec, valid_global_rec, valid_global_fm, valid_global_iou
