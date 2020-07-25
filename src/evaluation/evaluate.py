import sys
import os
import argparse

import numpy as np
from scipy.interpolate import interp1d
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from src import geometry, utils


def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


class PRCurve:
    def __init__(self, t_thres, R_thres, model):
        self.model = model
        self.t = t_thres
        self.R = R_thres
        self.precisions = None
        self.recalls = None
        self.scores = None

    def generate_curve(self, truth, proposals, scores, interpolate=True):
        """
        Generates PR curves given true query and proposal poses.
        Can select interpolation of precision, where precision values
        are replaced with maximum precision for all recall values
        greater or equal to current recall.
        """
        t_errs, R_errs = self._compute_errors(truth, proposals)
        scores_u = np.unique(scores)
        max_score = np.max(scores_u)
        self.scores = np.linspace(
            np.min(scores_u) - 1e-3, max_score + 1e-3, endpoint=True, num=1000
        )
        if self.model in ["Single", "Seq Match", "Graph"]:
            # ensures iterating through list means higher
            # model confidence
            self.scores = np.flip(self.scores)
        self.precisions = np.ones_like(self.scores)
        self.recalls = np.zeros_like(self.scores)
        self.F1 = np.zeros_like(self.scores)
        for i, score_thres in enumerate(self.scores):
            if self.model in ["Seq Match", "Single"]:
                localized = scores < score_thres
                t_err = t_errs
                R_err = R_errs
            else:
                ind_loc = self._localize_indices(scores, score_thres)
                # identify traverses where threshold met
                localized = ind_loc != -1
                t_err = np.squeeze(
                    np.take_along_axis(t_errs, ind_loc[:, np.newaxis], 1)
                )
                R_err = np.squeeze(
                    np.take_along_axis(R_errs, ind_loc[:, np.newaxis], 1)
                )
            correct = np.logical_and(t_err < self.t, R_err < self.R)
            # only count traverses with a proposal
            correct = np.logical_and(correct, localized)
            # compute precision and recall
            # index of -1 means not localized in max seq len
            nLocalized = np.count_nonzero(localized)
            nCorrect = np.count_nonzero(correct)
            if nLocalized > 0:
                # if none localized, precision = 1 by default
                self.precisions[i] = nCorrect / nLocalized
                if nCorrect + len(localized) - nLocalized > 0:
                    self.recalls[i] = nCorrect / (
                        nCorrect + len(localized) - nLocalized
                    )
        # flip curves for increasing recall
        self.precisions = np.flip(self.precisions)
        self.recalls = np.flip(self.recalls)
        self.scores = np.flip(self.scores)
        # ensure recalls are nondecreasing
        self.recalls, inds = np.unique(self.recalls, return_index=True)
        self.precisions = self.precisions[inds]
        self.scores = self.scores[inds]
        # chop off curve when recall first reaches 1
        ind_min = np.min(np.argwhere(self.recalls >= 1.0))
        self.recalls = self.recalls[: ind_min + 1]
        self.precisions = self.precisions[: ind_min + 1]
        self.scores = self.scores[: ind_min + 1]
        # interpolate precision, take max precision for
        # recall greater than raw recall
        if interpolate:
            for i in range(len(self.precisions)):
                self.precisions[i] = np.max(self.precisions[i:])
        return None

    def score_precision_level(self, pLevel):
        ind = np.argmax(np.argwhere(self.precisions >= pLevel))
        score = self.scores[ind]
        recall = self.recalls[ind]
        return score, recall

    def interpolate_at_recalls(self, recalls):
        """
        To ensure the same recall values in the PR curve
        (for intervals in MCL methods),
        interpolate existing curve at given recall values
        """
        f1 = interp1d(
            self.recalls, self.precisions, kind="linear", fill_value="extrapolate"
        )
        f2 = interp1d(
            self.recalls, self.scores, kind="linear", fill_value="extrapolate"
        )
        self.precisions = f1(recalls)
        self.scores = f2(recalls)
        self.recalls = recalls
        return None

    def auc(self):
        return metrics.auc(self.recalls, self.precisions)

    def _compute_errors(self, truth, proposals):
        if self.model not in ["Seq Match", "Single"]:
            # extract relative poses and distances
            t_errs = np.empty((len(truth), len(truth[0])))
            R_errs = t_errs.copy()
            for i in range(len(truth)):
                # extract translation and rotation distances separately
                rel_pose = truth[i].inv() * proposals[i]
                t_errs[i, :] = np.linalg.norm(rel_pose.t(), axis=1)
                R_errs[i, :] = rel_pose.R().magnitude()
        else:
            Trel = geometry.combine(truth) / geometry.combine(proposals)
            t_errs = np.linalg.norm(Trel.t(), axis=1)
            R_errs = Trel.R().magnitude()
        return t_errs, R_errs

    def _localize_indices(self, scores, score_thres):
        """
        For filtering methods, return indices where score passes
        threshold and -1 if threshold is never passed.
        For template matching methods, return 1 for localized or -1
        for not localized.
        """
        if self.model in ["Seq Match", "Single"]:
            # identify localization proposals when score is
            # above/below threshold
            not_loc = scores >= score_thres
            localized = np.ones(len(scores), dtype=int)
            localized[not_loc] = -1
        else:
            # identify localization proposals when score is
            # above threshold
            if self.model == "Graph":
                scores_loc = scores <= score_thres
            else:
                scores_loc = scores >= score_thres
            # take first localization in traverse and check pose error
            localized = first_nonzero(scores_loc, 1)
        return localized


def max_min_avg_PR(PR_curve_list):
    """
    Return upper bound, lower bound and mean PR curve from list.
    Assumes that each PR curve has same set of recall points (aligned).
    """
    precisions = []
    scores = []
    for PR in PR_curve_list:
        precisions.append(PR.precisions)
        scores.append(PR.scores)
    # find max and min possible precisions for confidence region
    precisions = np.asarray(precisions)
    scores = np.asarray(scores)
    max_prec = precisions.max(axis=0)
    min_prec = precisions.min(axis=0)
    mean_prec = precisions.mean(axis=0)
    # create PR curve objects
    PR_mean = PRCurve(PR.t, PR.R, PR.model)
    PR_mean.precisions = mean_prec
    PR_mean.recalls = PR.recalls
    PR_mean.scores = scores.mean(axis=0)

    PR_min = PRCurve(PR.t, PR.R, PR.model)
    PR_min.precisions = min_prec
    PR_min.recalls = PR.recalls

    PR_max = PRCurve(PR.t, PR.R, PR.model)
    PR_max.precisions = max_prec
    PR_max.recalls = PR.recalls
    return PR_mean, PR_min, PR_max
