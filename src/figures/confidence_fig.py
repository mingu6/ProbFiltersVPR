
import os
import argparse
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt

from src import geometry, utils, params
from src.models.TopologicalFilter import TopologicalFilter
from src.models.SingleImageMatching import SingleImageMatching
from src.thirdparty.nigh import Nigh

xlims = [619926, 620085]
ylims = [5735820, 5735900]


def main(args):
    # import reference and query traverses
    ref_poses, ref_descriptors, _ = \
        utils.import_reference_map(args.reference_traverse)
    query_poses, vo, _, query_descriptors, _ = \
        utils.import_query_traverse(args.query_traverse)
    # shortcuts used when plotting
    ref_poses_x, ref_poses_y = ref_poses.t()[:, 1], ref_poses.t()[:, 0]
    query_descriptors = query_descriptors[args.descriptor][args.nloc]
    query_poses = query_poses[args.nloc]
    query_poses_x, query_poses_y = \
        query_poses.t()[:, 1], query_poses.t()[:, 0]
    L = len(vo[args.nloc]) + 1  # max sequence length
    vo = vo[args.nloc]
    # setup topological filter
    topofilter = TopologicalFilter(ref_poses, ref_descriptors[args.descriptor],
                                   args.delta_topo,
                                   window_lower=args.window_lower,
                                   window_upper=args.window_upper)
    topofilter.initialize_model(query_descriptors[0, :])
    # setup single image matching
    single = SingleImageMatching(ref_poses, ref_descriptors[args.descriptor])

    # initialize plots
    fig = plt.figure(figsize=(14, 4))
    gs = fig.add_gridspec(1, 2, wspace=0, hspace=0)
    axs = []
    for i in range(2):
        axs.append(fig.add_subplot(gs[:, i]))
    for i in range(len(axs)):
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_aspect('equal')
        axs[i].set_xlim(xlims[0], xlims[1])
        axs[i].set_ylim(ylims[0], ylims[1])

    for t in range(L):
        if t > 0:
            topofilter.update(query_descriptors[t, :])
        proposalT, scoreT = topofilter.localize()
        proposalS, scoreS = single.localize_descriptor(query_descriptors[t, :])
        # visualize initial
        if t == 0:
            axs[0].scatter(ref_poses_x, ref_poses_y, c=topofilter.belief,
                           cmap='OrRd', s=12, vmin=0, vmax=3e-4)
            axs[0].set_title(
                r"t = {}, $\tau_t$ = {:.2f}".format(t, scoreT), fontsize=36
            )
            axs[1].scatter([proposalT.t()[1]], [proposalT.t()[0]],
                           color='fuchsia', s=400, marker='X')
            axs[0].scatter([query_poses_x[t]], [query_poses_y[t]],
                           color='limegreen', s=600, marker='*')
        # visualize intermediate
        if t == args.niter:
            axs[1].scatter(ref_poses_x, ref_poses_y, c=topofilter.belief,
                           cmap='OrRd', s=12, vmin=0, vmax=4e-3)
            axs[1].set_title(r"t = {}, $\tau_t$ = {:.2f}".format(t, scoreT),
                             fontsize=36)
            axs[1].scatter([proposalT.t()[1]], [proposalT.t()[0]],
                           color='fuchsia', s=400, marker='X', alpha=1.0)
            axs[1].scatter([query_poses_x[t]], [query_poses_y[t]],
                           color='limegreen', s=600, marker='*', alpha=0.8)

    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(utils.figures_path + '/confidence_fig.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate first system figure for paper")
    parser.add_argument('-r', '--reference-traverse',
                        type=str, default='Overcast',
                        help="reference traverse used to build map")
    parser.add_argument('-q', '--query-traverse', type=str, default='Night',
                        help="query traverse to localize against"
                        "reference map")
    parser.add_argument('-d', '--descriptor', type=str, default='NetVLAD',
                        help='descriptor type to run model using.')
    parser.add_argument('-nl', '--nloc', type=int, default=5,
                        help="which trial to generate figure for")
    parser.add_argument('-ni', '--niter', type=int, default=4,
                        help="which intermediate iteration to check figure")
    # topological filter parameters
    parser.add_argument('-dt', '--delta-topo', type=float, default=5,
                        help="multiple used for calibrating sensor update"
                        "rate parameter. assumes 6sigma change in image"
                        "difference causes m times sensor update")
    parser.add_argument('-wl', '--window-lower', type=int, default=-2,
                        help="minimum state transition in transition matrix")
    parser.add_argument('-wu', '--window-upper', type=int, default=10,
                        help="maximum state transition in transition matrix")
    args = parser.parse_args()
    main(args)
