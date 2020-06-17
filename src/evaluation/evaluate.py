import sys
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from src import geometry

################## filepaths ###################
file_dir = os.path.dirname(os.path.abspath(__file__))
raw_path = os.path.abspath(os.path.join(file_dir, "../../", "data/raw/RobotCar/"))
processed_path = os.path.abspath(os.path.join(file_dir, "../../", "data/processed/RobotCar/"))
results_path = os.path.abspath(os.path.join(file_dir, "../../", "results"))
################################################
def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def extract_gt_dists(query_gt, proposals):
    # extract relative poses and distances
    t_dists = np.empty((len(query_gt), len(query_gt[0])))
    R_dists = t_dists.copy()
    for i in range(len(query_gt)):
        # extract translation and rotation distances separately
        rel_pose = query_gt[i].inv() * proposals[i]
        t_dists[i, :] = np.linalg.norm(rel_pose.t(), axis=1)
        R_dists[i, :] = rel_pose.R().magnitude()
    return t_dists, R_dists

def localize_indices(scores, score_thres, model):
    # identify localization proposals when score is above/below threshold
    if 'Particle' in model or model == 'Odometry Only' or model == 'SeqSLAM':
        scores_loc = scores < score_thres
    else:
        scores_loc = scores > score_thres
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
    if 'HMM' not in model:
        scores_u = np.flip(scores_u) # ensures iterating through list means higher model confidence
    # scores_u = scores_u[7500:] # night
    # scores_u = scores_u[7000:]
    # scores_u = scores_u[7000:]
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

def main(args):
    deg = np.pi / 180
    t_thresholds = np.array([3., 5.])
    R_thresholds = np.array([5., 10.])
    # R_thresholds = np.array([180., 180.])
    # load result data
    results_path1 = os.path.join(results_path, args.traverse)
    results = [f for f in os.listdir(results_path1) if f.endswith(".pickle")]
    # initialize plots for PR curves
    fig_pr, axs_pr = plt.subplots(1, len(t_thresholds), figsize=(10, 5))
    for ax, t_t, R_t in zip(axs_pr, t_thresholds, R_thresholds):
        ax.set_title('Tolerance: {:.0f}m {:.0f}deg'.format(t_t, R_t))
        # ax.set_xlabel('Recall')
        # ax.set_ylabel('Precision')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.1)
        ax.set_aspect('equal')
    # load a single file and extract seq len
    fpath = os.path.abspath(os.path.join(results_path1, results[0]))
    with open(fpath, 'rb') as f:
        result = pickle.load(f)
    nloc = len(result['query_gt'])
    # save max recall at precision thresholds
    max_recall_data_99 = np.empty((len(results), len(t_thresholds)))
    max_recall_data_100 = np.empty((len(results), len(t_thresholds)))
    max_recall_data_99[:] = np.nan
    max_recall_data_100[:] = np.nan
    # save compute time to localization at max recall for fixed prec.
    compute_times_99 = np.empty((len(results), len(t_thresholds), nloc)) 
    compute_times_100 = np.empty((len(results), len(t_thresholds), nloc))
    compute_times_99[:] = np.nan
    compute_times_100[:] = np.nan
    # save number of time steps to localization at max recall for fixed prec.
    num_steps_99 = np.zeros((len(results), len(t_thresholds), nloc))
    num_steps_100 = np.zeros((len(results), len(t_thresholds), nloc))
    num_steps_99[:] = np.nan
    num_steps_100[:] = np.nan
    # save metric error of each run at max recall for fixed prec.
    errt_data_99 = np.zeros((len(results), len(t_thresholds), nloc))
    errt_data_100 = np.zeros((len(results), len(t_thresholds), nloc))
    errR_data_99 = np.zeros((len(results), len(t_thresholds), nloc))
    errR_data_100 = np.zeros((len(results), len(t_thresholds), nloc))
    errt_data_99[:] = np.nan
    errt_data_100[:] = np.nan
    errR_data_99[:] = np.nan
    errR_data_100[:] = np.nan

    model_names = []
    scores_99_all = np.zeros((len(results), len(t_thresholds)))
    scores_100_all = np.zeros((len(results), len(t_thresholds)))

    for i, fname in enumerate(tqdm(results)):
        fpath = os.path.abspath(os.path.join(results_path1, fname))
        with open(fpath, 'rb') as f:
            result = pickle.load(f)
        if result['model'] not in ['SeqSLAM', 'Single']:
            t_dists, R_dists = extract_gt_dists(result['query_gt'], result['proposals'])
        else:
            Trel = result['query_gt'] / result['proposals']
            t_dists = np.linalg.norm(Trel.t(), axis=1)
            R_dists = Trel.R().magnitude()

        for j, (ax_pr, t_t, R_t) in enumerate(zip(axs_pr, t_thresholds, R_thresholds)):
            if result['model'] not in model_names:
                model_names.append(result['model'])
            # plot PR curve
            if result['model'] not in ['SeqSLAM', 'Single']:
                precisions, recalls, scores = generate_pr_curve(result['scores'], t_dists, R_dists, 
                                            t_t, R_t * deg, result['model'])
            else:
                precisions, recalls, scores = generate_pr_curve_single(result['scores'], t_dists, R_dists, 
                                            t_t, R_t * deg)
            ax_pr.plot(recalls, precisions, label=result['model'])
            # identify max recall at precision levels and plot
            ind_99 = np.argwhere(precisions > 0.99)
            ind_max_99 = np.squeeze(ind_99[np.argmax(recalls[ind_99])]) # identify score with highest recall for given precision threshold
            ind_100 = np.argwhere(precisions == 1.0)
            ind_max_100 = np.squeeze(ind_99[np.argmax(recalls[ind_100])]) # identify score with highest recall for given precision threshold
            max_recall_data_99[i, j] = recalls[ind_max_99]
            max_recall_data_100[i, j] = recalls[ind_max_100]
            # scores (100% and 99% precision)
            scores_99 = scores[ind_max_99] # optimal score for max recall at precision level
            scores_100 = scores[ind_max_100]
            scores_99_all[i, j] = scores_99
            scores_100_all[i, j] = scores_100
            if result['model'] not in ['SeqSLAM', 'Single']:
                loc_ind_99 = localize_indices(result['scores'], scores_99, result['model'])
                loc_ind_100 = localize_indices(result['scores'], scores_100, result['model'])
            else:
                loc_ind_99 = localize_indices_single(result['scores'], scores_99)
                loc_ind_100 = localize_indices_single(result['scores'], scores_100)
            compute_times = np.asarray(result['times'])
            if result['model'] not in ['SeqSLAM', 'Single']:
                compute_at_ind_99 = np.squeeze(np.take_along_axis(compute_times, loc_ind_99[:, np.newaxis], 1))
                compute_at_ind_100 = np.squeeze(np.take_along_axis(compute_times, loc_ind_100[:, np.newaxis], 1))
            else:
                compute_at_ind_99 = compute_times
                compute_at_ind_100 = compute_times
            compute_times_99[i, j, loc_ind_99 != -1] = compute_at_ind_99[loc_ind_99 != -1]
            compute_times_100[i, j, loc_ind_100 != -1] = compute_at_ind_100[loc_ind_100 != -1]
            # number of steps to localize for fixed thres (100% and 99% precision)
            if result['model'] not in ['SeqSLAM', 'Single']:
                steps = np.arange(len(result['query_gt'][0]))
                num_steps_99[i, j, loc_ind_99 != -1] = steps[loc_ind_99][loc_ind_99 != -1]
                num_steps_100[i, j, loc_ind_100 != -1] = steps[loc_ind_100][loc_ind_100 != -1]
            elif result['model'] == 'SeqSLAM':
                num_steps_99[i, j, :] = result['L']
                num_steps_100[i, j, :] = result['L']
            else:
                num_steps_99[i, j, :] = 1
                num_steps_100[i, j, :] = 1 # single imagematching
            if result['model'] not in ['SeqSLAM', 'Single']:
                # metric errors for thresh (100% and 99% precision)
                errt_at_99 = np.squeeze(np.take_along_axis(t_dists, loc_ind_99[:, np.newaxis], 1))
                errt_at_100 = np.squeeze(np.take_along_axis(t_dists, loc_ind_100[:, np.newaxis], 1))
                errR_at_99 = np.squeeze(np.take_along_axis(R_dists, loc_ind_99[:, np.newaxis], 1))
                errR_at_100 = np.squeeze(np.take_along_axis(R_dists, loc_ind_100[:, np.newaxis], 1))
            else:
                errt_at_99 = t_dists
                errt_at_100 = t_dists
                errR_at_99 = R_dists
                errR_at_100 = R_dists
            errt_data_99[i, j, loc_ind_99 != -1] = errt_at_99[loc_ind_99 != -1]
            errt_data_100[i, j, loc_ind_100 != -1] = errt_at_100[loc_ind_100 != -1]
            errR_data_99[i, j, loc_ind_99 != -1] = errR_at_99[loc_ind_99 != -1]
            errR_data_100[i, j, loc_ind_100 != -1] = errR_at_100[loc_ind_100 != -1]

    # max recall at fixed precision
    print("Model names", model_names)
    print("translation thresholds (m)", t_thresholds)
    print("Rotation thresholds (deg)", R_thresholds)
    print("Max recall at 99% precision")
    print(max_recall_data_99)
    print("Max recall at 100% precision")
    print(max_recall_data_100)
    print("Scores at 99% precision")
    print(scores_99_all)
    print("Scores at 100% precision")
    print(scores_100_all)
    # compute times until localized for fixed precision
    print("Total compute time until localization at 99% precision")
    print("mean\n", np.nanmean(compute_times_99, axis=2), "\nstd dev:\n", np.nanstd(compute_times_99, axis=2))
    print("Total compute time until localization at 100% precision")
    print("mean\n", np.nanmean(compute_times_100, axis=2), "\nstd dev:\n", np.nanstd(compute_times_100, axis=2))
    # number of steps until localized
    print("Number of steps until localized at 99% precision")
    print("mean\n", np.nanmean(num_steps_99, axis=2), "\nstd dev:\n", np.nanstd(num_steps_99, axis=2))
    print("Number of steps until localized at 100% precision")
    print("mean\n", np.nanmean(num_steps_100, axis=2), "\nstd dev:\n", np.nanstd(num_steps_100, axis=2))
    # time per iteration until localization
    print("Time per iteration until localized at 99% precision")
    print("mean\n", np.nanmean(compute_times_99 / num_steps_99, axis=2), "\nstd dev:\n", np.nanstd(compute_times_99 / num_steps_99, axis=2))
    print("Time per iteration until localized at 100% precision")
    print("mean\n", np.nanmean(compute_times_100 / num_steps_100, axis=2), "\nstd dev:\n", np.nanstd(compute_times_100 / num_steps_100, axis=2))
    # metric errors
    print("Metric error of proposals at 99% precision, translation (m)")
    print("mean\n", np.nanmean(errt_data_99, axis=2), "\nstd dev:\n", np.nanstd(errt_data_99, axis=2))
    print("Metric error of proposals at 100% precision, translation (m)")
    print("mean\n", np.nanmean(errt_data_100, axis=2), "\nstd dev:\n", np.nanstd(errt_data_100, axis=2))
    print("Metric error of proposals at 99% precision, rotation (deg)")
    print("mean\n", np.nanmean(errR_data_99, axis=2) / deg, "\nstd dev:\n", np.nanstd(errR_data_99, axis=2) / deg)
    print("Metric error of proposals at 100% precision, rotation (deg)")
    print("mean\n", np.nanmean(errR_data_100, axis=2) / deg, "\nstd dev:\n", np.nanstd(errR_data_100, axis=2) / deg)

    # plot PR curve
    # handles, labels = ax_pr.get_legend_handles_labels()
    # fig_pr.legend(handles, labels, loc='lower center')
    axs_pr[-1].legend(loc='lower right')
    fig_pr.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    fig_pr.suptitle(args.traverse)
    fig_pr.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('Recall')
    # plt.ylabel('Precision')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plot performance curves using saved run data")
    parser.add_argument('-t', '--traverse', type=str, required=True, choices=['Rain', 'Dusk', 'Night'],
                        help="which query traverse to plot")
    args = parser.parse_args()
    main(args)
