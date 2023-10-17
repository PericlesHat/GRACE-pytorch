# ---- coding: utf-8 ----
# @author: Carl Yang
# @description: Evaluation metrics, from original grace code

def f1_pair(pred_y, true_y):
    """calculate f1 score for a pair of communities (predicted and ground truth)

    args:
        pred_y (N * 1): binary array, 1 means the corresponding instance belongs to predicted community
        true_y (N * 1): binary array, 1 means the corresponding instance belongs to golden community
    """
    corrected = sum([2 == (py + ty) for (py, ty) in zip(pred_y, true_y)])
    if 0 == corrected:
        return 0, 0, 0
    precision = float(corrected) / sum(pred_y)
    recall = float(corrected) / sum(true_y)
    f1score = 2 * precision * recall / (precision + recall)
    return f1score, precision, recall

def f1_community(pre_ys, true_ys):
    """calculate f1 score for two sets of communities (predicted and ground truth)

    args:
        pred_ys (k * N):
        true_ys (l * N):
    """
    tot_size = 0
    tot_fscore = 0.0
    for pre_y in pre_ys:
        cur_size = sum(pre_y)
        tot_size += cur_size
        tot_fscore += max([f1_pair(pre_y, true_y)[0] for true_y in true_ys]) * cur_size
    return float(tot_fscore) / tot_size

def f1_community(pre_ys, true_ys):
    """calculate f1 score for two sets of communities (predicted and ground truth)

    args:
        pred_ys (k * N):
        true_ys (l * N):
    """
    tot_size = 0
    tot_fscore = 0.0
    for pre_y in pre_ys:
        cur_size = sum(pre_y)
        tot_size += cur_size
        tot_fscore += max([f1_pair(pre_y, true_y)[0] for true_y in true_ys]) * cur_size
    return float(tot_fscore) / tot_size


def jc_pair(pred_y, true_y):
    """calculate jc score for a pair of communities (predicted and ground truth)

    args:
        pred_y (N * 1): binary array, 1 means the corresponding instance belongs to predicted community
        true_y (N * 1): binary array, 1 means the corresponding instance belongs to golden community
    """
    corrected = sum([ 2 == (py + ty) for (py, ty) in zip(pred_y, true_y)])
    if 0 == corrected:
        return 0
    tot = sum([ (py + ty) > 0 for (py, ty) in zip(pred_y, true_y)])
    return float(corrected) / tot

def jc_community(pre_ys, true_ys):
    """calculate jc score for two sets of communities (predicted and ground truth)

    args:
        pred_ys (k * N):
        true_ys (l * N):
    """
    tot_jcscore = 0.0

    tmp_size = float(1) / ( len(pre_ys) * 2 )
    for pre_y in pre_ys:
        tot_jcscore += max([jc_pair(pre_y, true_y) for true_y in true_ys]) * tmp_size

    tmp_size = float(1) / ( len(true_ys) * 2 )
    for true_y in true_ys:
        tot_jcscore += max([jc_pair(pre_y, true_y) for pre_y in pre_ys]) * tmp_size

    return tot_jcscore