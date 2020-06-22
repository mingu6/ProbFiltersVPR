import sys
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from src import geometry, utils

def extract_gt_dists(query_gt, proposals, model):
    if model not in ['SeqSLAM', 'Single']:
        # extract relative poses and distances
        t_dists = np.empty((len(query_gt), len(query_gt[0])))
        R_dists = t_dists.copy()
        for i in range(len(query_gt)):
            # extract translation and rotation distances separately
            rel_pose = query_gt[i].inv() * proposals[i]
            t_dists[i, :] = np.linalg.norm(rel_pose.t(), axis=1)
            R_dists[i, :] = rel_pose.R().magnitude()
    else:
        Trel = geometry.combine(query_gt) / geometry.combine(proposals)
        t_dists = np.linalg.norm(Trel.t(), axis=1)
        R_dists = Trel.R().magnitude()
    return t_dists, R_dists

def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def localize_indices(scores, score_thres, model):
    # identify localization proposals when score is above/below threshold
    if model == 'Topological':
        scores_loc = scores > score_thres
    else:
        scores_loc = scores < score_thres
    # take first localization in traverse and check pose error
    ind_loc = first_nonzero(scores_loc, 1)
    return ind_loc

def localize_indices_single(scores, score_thres):
    # identify localization proposals when score is above/below threshold
    not_loc = scores > score_thres
    localized = np.ones(len(scores))
    localized[not_loc] = -1
    return localized

def generate_pr_curve(scores, t_dists, R_dists, t_thres, R_thres, model):
    scores_u = np.unique(scores)
    if model != 'Topological':
        scores_u = np.flip(scores_u) # ensures iterating through list means higher model confidence
    scores_u = scores_u[int(len(scores_u) * 0.5):] # chop off tail
    precisions = np.ones_like(scores_u)
    recalls = np.empty_like(scores_u)
    for i, score_thres in enumerate(scores_u):
        ind_loc = localize_indices(scores, score_thres, model)
        loc_err_t = np.squeeze(np.take_along_axis(t_dists, ind_loc[:, np.newaxis], 1))
        loc_err_R = np.squeeze(np.take_along_axis(R_dists, ind_loc[:, np.newaxis], 1))
        correct = np.logical_and(loc_err_t < t_thres, loc_err_R < R_thres)
        correct = np.logical_and(correct, ind_loc != -1) # only count traverses with a proposal
        # compute precision and recall
        n_localized = np.count_nonzero(ind_loc != -1) # index of -1 means not localized in max seq len
        n_correct = np.count_nonzero(correct)
        if n_localized > 0:
            precisions[i] = n_correct / n_localized # if none localized, precision = 1 by default
        recalls[i] = n_correct / len(correct)
    return precisions, recalls, scores_u

def generate_pr_curve_single(scores, t_dists, R_dists, t_thres, R_thres):
    scores_u = np.unique(scores)
    scores_u = np.flip(scores_u) # lower scores indicate higher confidence
    precisions = np.ones_like(scores_u)
    recalls = np.empty_like(scores_u)
    for i, score_thres in enumerate(scores_u):
        localized = scores < score_thres
        correct = np.logical_and(t_dists < t_thres, R_dists < R_thres)
        correct = np.logical_and(correct, localized) # only count traverses with a proposal
        # compute precision and recall
        nLocalized = np.count_nonzero(localized)
        nCorrect = np.count_nonzero(correct)
        if nLocalized > 0:
            precisions[i] = nCorrect / nLocalized # if none localized, precision = 1 by default
        recalls[i] = nCorrect / len(correct)
    return precisions, recalls, scores_u

def score_precision_level(precisions, recalls, scores, pLevel):
    ind = np.argwhere(precisions >= pLevel)
    ind_max = np.squeeze(ind[np.argmax(recalls[ind])]) # identify score with highest recall for given precision threshold
    return scores[ind_max], recalls[ind_max]