import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pickle

from tabulate import tabulate

from src import utils, geometry
from src.evaluation import evaluate

def main(args):
    
    nQueries = len(args.query_traverses)
    nDescriptors = len(args.descriptors)
    results_path_0 = os.path.join(utils.results_path, args.query_traverses[0], args.descriptors[0])
    models = [model[:-7] for model in sorted(os.listdir(results_path_0))]
    nModels = len(models)

    # initialize results arrays
    max_recall_results = np.empty((nModels, nQueries, nDescriptors))
    compute_results = np.empty((nModels, nQueries, nDescriptors))
    nsteps_results = np.empty((nModels, nQueries, nDescriptors)) 

    for i in range(nQueries):
        for j in range(nDescriptors):
            results_path = os.path.join(utils.results_path, args.query_traverses[i], args.descriptors[j])
            for k, fname in enumerate(sorted(os.listdir(results_path))):
                with open(os.path.join(results_path, fname), 'rb') as f:
                    results = pickle.load(f)
                # extract results data from dict
                model = results['model']
                query_poses = results['query_gt']
                proposals = results['proposals']
                scores = results['scores']
                times = np.asarray(results['times'])

                # generate PR curve data for max recall table
                t_dists, R_dists = evaluate.extract_gt_dists(query_poses, proposals, model)
                if model not in ['SeqSLAM', 'Single']:
                    precisions, recalls, scores_pr = evaluate.generate_pr_curve(scores, t_dists, R_dists, 
                                                args.t_thres, args.R_thres * np.pi / 180, model)
                else:
                    precisions, recalls, scores_pr = evaluate.generate_pr_curve_single(scores, t_dists, R_dists, 
                                                args.t_thres, args.R_thres * np.pi / 180)
                score, recall = evaluate.score_precision_level(precisions, recalls, scores_pr, args.precision_level)
                max_recall_results[k, i, j] = recall * 100

                # evaluate average compute time and number of steps
                if model not in ['SeqSLAM', 'Single']:
                    step_localized = evaluate.localize_indices(scores, score, model)
                    compute_at_localized = np.squeeze(np.take_along_axis(times, step_localized[:, np.newaxis], 1))
                    total_compute = compute_at_localized[step_localized != -1]
                    nsteps = step_localized[step_localized != -1]
                else:
                    localized = evaluate.localize_indices_single(scores, score)
                    total_compute = times[localized != 1]
                    nsteps = np.ones_like(total_compute)
                    if model == 'SeqSLAM':
                        # L = results['L']
                        L = 30
                        nsteps *= L
                compute_results[k, i, j] = np.mean(total_compute * 1000 / nsteps) # average time per iteration
                nsteps_results[k, i, j] = np.mean(nsteps)
    # create tables using results
    table_maxrecall = []
    table_compute = []
    table_nsteps = []
    for i in range(nModels):
        # create table rows
        table_maxrecall.append([models[i], *max_recall_results[i].reshape(-1).tolist()])
        table_compute.append([models[i], *compute_results[i].reshape(-1).tolist()])
        table_nsteps.append([models[i], *nsteps_results[i].reshape(-1).tolist()])

    headers = ["Models"]
    for i in range(nQueries):
        for j in range(nDescriptors):
            headers.append("{} {}".format(args.query_traverses[i], args.descriptors[j]))
    fname_out = os.path.join(utils.figures_path, args.filename)
    print("Tolerance: {}m {}deg\n".format(args.t_thres, args.R_thres), file=open(fname_out, "w"))
    print("Max recall at {}\n\n".format(args.precision_level), tabulate(table_maxrecall, floatfmt=".1f", headers=headers), file=open(fname_out, "a"))
    print("\nCompute time per iteration\n\n", tabulate(table_compute, floatfmt=".1f", headers=headers), file=open(fname_out, "a"))
    print("\nNumber of steps until localized\n\n", tabulate(table_nsteps, floatfmt=".1f", headers=headers), file=open(fname_out, "a"))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SeqSLAM on trials")
    parser.add_argument('-q', '--query-traverses', nargs='+', type=str, default=['Rain', 'Dusk', 'Night'],
                        help="Names of query traverses to plot results for e.g. Overcast, Night, Dusk etc.")
    parser.add_argument('-d', '--descriptors', nargs='+', type=str, default=['NetVLAD', 'DenseVLAD'], help='descriptor types to run experiments on.')
    parser.add_argument('-t', '--t-thres', type=float, default=5, 
        help="maximum error for translation distance (m) for a correct localization")
    parser.add_argument('-R', '--R-thres', type=float, default=10, 
        help="maximum error for angle (degrees) for a correct localization")
    parser.add_argument('-p', '--precision-level', type=float, default=0.99, 
        help="Precision level to evaluate table elements at")
    parser.add_argument('-f', '--filename', type=str, default='tables.txt', 
        help="Filename for output")
    args = parser.parse_args()

    main(args)