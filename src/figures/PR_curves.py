import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pickle

from src import utils, geometry
from src.evaluation import evaluate

def main(args):
    
    nQueries = len(args.query_traverses)
    nDescriptors = len(args.descriptors)
    fig, axs = plt.subplots(nDescriptors, nQueries, figsize=(6 * nQueries, 4.2 * nDescriptors))
    if nDescriptors == 1:
        axs = [axs]

    for i in range(nDescriptors):
        for j in range(nQueries):
            results_path = os.path.join(utils.results_path, args.query_traverses[j], args.descriptors[i])
            for fname in sorted(os.listdir(results_path)):
                with open(os.path.join(results_path, fname), 'rb') as f:
                    results = pickle.load(f)
                # extract results data from dict
                model = results['model']
                query_poses = results['query_gt']
                proposals = results['proposals']
                scores = results['scores']

                t_dists, R_dists = evaluate.extract_gt_dists(query_poses, proposals, model)
                # initialize plots
                axs[i][j].set_aspect(0.6)
                if i == 0:
                    axs[i][j].set_title(args.query_traverses[j], fontsize=20)
                axs[i][j].set_xlabel('Recall', fontsize=14)
                axs[i][j].set_ylabel('Precision', fontsize=14)
                axs[i][j].set_xlim(0, 1)
                axs[i][j].set_ylim(0, 1.1)
                # generate PR curve data using scores, metric error and thresholds                    
                if model not in ['SeqSLAM', 'Single']:
                    precisions, recalls, scores = evaluate.generate_pr_curve(scores, t_dists, R_dists, 
                                                args.t_thres, args.R_thres * np.pi / 180, model)
                else:
                    precisions, recalls, scores = evaluate.generate_pr_curve_single(scores, t_dists, R_dists, 
                                                args.t_thres, args.R_thres * np.pi / 180)

                # plot
                axs[i][j].plot(recalls, precisions, label=model)
    axs[-1][-1].legend(loc='lower right', fontsize=12)
    # fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    fig.tight_layout()
    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(utils.figures_path + '/PR_{:.0f}m_{:.0f}d.png'.format(args.t_thres, args.R_thres), bbox_inches='tight')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SeqSLAM on trials")
    parser.add_argument('-q', '--query-traverses', nargs='+', type=str, default=['Rain', 'Dusk', 'Night'],
                        help="Names of query traverses to plot results for e.g. Overcast, Night, Dusk etc.")
    parser.add_argument('-d', '--descriptors', nargs='+', type=str, default=['NetVLAD', 'DenseVLAD'], help='descriptor types to run experiments on.')
    parser.add_argument('-t', '--t-thres', type=float, default=5, 
        help="maximum error for translation distance (m) for a correct localization")
    parser.add_argument('-R', '--R-thres', type=float, default=10, 
        help="maximum error for angle (degrees) for a correct localization")
    args = parser.parse_args()

    main(args)