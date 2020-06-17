
import os
import argparse
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt

from src.data.RobotCar.reference_traverse import save_obj
from src.models.VPR_MCL import PlaceRecFilter
from src.models.run_VPR_MCL import import_traverses
from src.models.HMM_baseline import PlaceRecHMM
from src import geometry
from src.models import params
from thirdparty.nigh.Nigh import SE3Tree

################## filepaths ###################
file_dir = os.path.dirname(os.path.abspath(__file__))
raw_path = os.path.abspath(os.path.join(file_dir, "../../", "data/raw/RobotCar/"))
processed_path = os.path.abspath(os.path.join(file_dir, "../../", "data/processed/RobotCar/"))
save_path = os.path.abspath(os.path.join(file_dir, "../../", "figures"))
################################################

def main(args):
    # import reference and query traverses
    ref_gt, ref_descriptors, ind, query_gt, vo, query_descriptors, ref_tstamps, query_tstamps, w\
        = import_traverses(args.reference_traverse, args.query_traverse, 'NetVLAD', gt_motion=True)
    ref_gt_x, ref_gt_y = ref_gt.t()[:, 1], ref_gt.t()[:, 0]
    query_descriptors = query_descriptors[args.nloc]
    query_gt = query_gt[args.nloc]
    query_gt_x, query_gt_y = query_gt.t()[:, 1], query_gt.t()[:, 0]
    L = len(vo[args.nloc]) + 1 # max sequence length
    vo = vo[args.nloc]
    # setup particle filter
    poses_tree = SE3Tree(2 * args.attitude_weight)
    poses_tree.insert(ref_gt.t(), ref_gt.R().as_quat())
    particleFilter = PlaceRecFilter(args.query_traverse, args.n_particles, args.rate_pose, args.k_pose, args.delta_particles, args.attitude_weight,
                poses_tree, ref_gt, ref_descriptors, query_descriptors[0, :], None, auxiliary=True)
    # setup topological filter
    topoFilter = PlaceRecHMM(args.delta_topo, ref_gt, ref_descriptors, query_descriptors[0, :], 
            trans_window=args.window_trans, trans_sigma=None, offset=args.offset, 
            filter_obs=None, obs_window=None, obs_sigma=None)

    # initialize plots
    fig = plt.figure(figsize=(20, 5))
    gs = fig.add_gridspec(2, 4, wspace=0, hspace=0)
    axs = []
    for i in range(4):
        if i < 3:
            axs.append(fig.add_subplot(gs[:, i]))
        else:
            axs.append(fig.add_subplot(gs[0, i]))
            axs.append(fig.add_subplot(gs[1, i]))
    for i in range(len(axs)):
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_aspect('equal', adjustable='datalim')
        # fixed axes limits for global view
        if i < 3:
            axs[i].set_xlim(np.min(ref_gt_x) - 100, np.max(ref_gt_x) + 100)
            axs[i].set_ylim(np.min(ref_gt_y) - 100, np.max(ref_gt_y) + 100)

    localizedT = False
    localizedP = False

    for t in range(L):
        if t == 0:
            proposalP, scoreP = particleFilter.localize_particles()
        else:
            proposalP, scoreP = particleFilter.monte_carlo_update(vo[t-1], query_descriptors[t, :])
            topoFilter.belief_update(query_descriptors[t, :])
        proposalT, scoreT = topoFilter.localize_belief()
        # visualize initial
        if t == 0:
            axs[0].scatter(ref_gt_x, ref_gt_y, c=topoFilter.belief, cmap='jet', s=8, vmin=0, vmax=2e-4)
            axs[0].set_title("t = {}, confidence = {:.2f}".format(t, scoreT))
        # visualize intermediate
        if t == args.niter:
            axs[1].scatter(ref_gt_x, ref_gt_y, c=topoFilter.belief, cmap='jet', s=8, vmin=0, vmax=2e-4)
            axs[1].set_title("t = {}, confidence = {:.2f}".format(t, scoreT))
        # visualize final
        if scoreT > 0.95 and not localizedT:
            axs[2].scatter(ref_gt_x, ref_gt_y, c=topoFilter.belief, cmap='jet', s=8, vmin=0, vmax=1e-2)
            axs[2].set_title("t = {}, confidence = {:.2f}".format(t, scoreT))
            # inlet topological
            axs[3].scatter(ref_gt_x, ref_gt_y, c=topoFilter.belief, cmap='jet', s=8, vmin=0, vmax=1e-1)
            axs[3].set_xlim(query_gt_x[t] - 10, query_gt_x[t] + 10)
            axs[3].set_ylim(query_gt_y[t] - 5, query_gt_y[t] + 5)
            axs[3].scatter([ref_gt[proposalT].t()[1]], [ref_gt[proposalT].t()[0]], color='purple', s=300, marker='*')
            axs[3].scatter([query_gt_x[t]], [query_gt_y[t]], color='green', s=200, marker='X')
            # set as localized
            localizedT = True
        # inlet PF
        if scoreP < 3 and not localizedP:
            axs[4].scatter(particleFilter.particles.t()[:, 1], particleFilter.particles.t()[:, 0], cmap='jet', c=particleFilter.weights, s=8)
            axs[4].scatter(ref_gt_x, ref_gt_y, color='black', s=8)
            axs[4].scatter([proposalP.t()[1]], [proposalP.t()[0]], color='purple', s=300, marker='*')
            axs[4].scatter([query_gt_x[t]], [query_gt_y[t]], color='green', s=200, marker='X')
            axs[4].set_xlim(query_gt_x[t] - 10, query_gt_x[t] + 10)
            axs[4].set_ylim(query_gt_y[t] - 5, query_gt_y[t] + 5)
            # set as localized
            localizedP = True
        
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(save_path + '/catchy_figure.png')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCL algorithm to localise")
    parser.add_argument('-r', '--reference-traverse', type=str, default='2015-03-17-11-08-44',
                        help="reference traverse used to build map")
    parser.add_argument('-q', '--query-traverse', type=str, default='2014-12-16-18-44-24',
                        help="query traverse to localize against reference map")
    parser.add_argument('-nl', '--nloc', type=int, default=311, 
        help="which trial to generate figure for")
    parser.add_argument('-ni', '--niter', type=int, default=10, 
        help="which intermediate iteration to check figure")
    # Particle filter parameters
    parser.add_argument('-m', '--n-particles', type=int, default=3000, 
        help="number of particles to use during localisation")
    parser.add_argument('-w', '--attitude-weight', type=float, default=20, 
        help="weight for attitude components of pose distance equal to 1 / d for d being rotation angle (rad) equivalent to 1m translation")
    parser.add_argument('-dp', '--delta-particles', type=float, default=20, 
        help="multiple used for calibrating sensor update rate parameter. assumes 6sigma change in image difference causes m times sensor update")
    parser.add_argument('-rp', '--rate-pose', type=float, default=0.2, 
        help="rate parameter for computing pose weights")
    parser.add_argument('-kp', '--k-pose', type=int, default=8, 
        help='number of nearest neighbours keyframes for each particle in observation likelihood')
    # Topological filter parameters
    parser.add_argument('-dt', '--delta-topo', type=float, default=20, 
        help="multiple used for calibrating sensor update rate parameter. assumes 6sigma change in image difference causes m times sensor update")
    parser.add_argument('-wt', '--window-trans', type=int, default=6, 
        help="size of window for possible transitions in frames")
    parser.add_argument('-o', '--offset', type=int, default=4, 
        help="mean translation between query observations")
    args = parser.parse_args()
    main(args)