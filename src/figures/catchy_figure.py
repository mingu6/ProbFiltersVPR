
import os
import argparse
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt

from src import geometry, utils, params
from src.models.ParticleFilter import ParticleFilter
from src.models.TopologicalFilter import TopologicalFilter
from src.models.SeqSLAM import SeqSLAM
from src.models.SingleImageMatching import SingleImageMatching
from src.thirdparty.nigh import Nigh

def main(args):
    # import reference and query traverses
    ref_poses, ref_descriptors, _ = utils.import_reference_map(args.reference_traverse)
    query_poses, vo, _, query_descriptors, _ = utils.import_query_traverse(args.query_traverse)
    # shortcuts used when plotting
    ref_poses_x, ref_poses_y = ref_poses.t()[:, 1], ref_poses.t()[:, 0]
    query_descriptors = query_descriptors[args.descriptor][args.nloc]
    query_poses = query_poses[args.nloc]
    query_poses_x, query_poses_y = query_poses.t()[:, 1], query_poses.t()[:, 0]
    L = len(vo[args.nloc]) + 1 # max sequence length
    vo = vo[args.nloc]
    # setup particle filter
    ref_tree = Nigh.SE3Tree(2 * args.attitude_weight) # 2 times is b/c of rotation angle representation in library
    ref_tree.insert(ref_poses.t(), ref_poses.R().as_quat())
    particlefilter = ParticleFilter(ref_tree, ref_poses, ref_descriptors[args.descriptor], args.nparticles, args.lambda2, args.k_pose,  
                        args.delta_particles, args.attitude_weight, params.sigma_init, params.sigma_vo[args.query_traverse])
    particlefilter.initialize_model(query_descriptors[0, :])
    # setup topological filter
    topofilter = TopologicalFilter(ref_poses, ref_descriptors[args.descriptor], args.delta_topo, window_lower=args.window_lower, window_upper=args.window_upper)
    topofilter.initialize_model(query_descriptors[0, :])
    # setup and localize SeqSLAM
    seqslam = SeqSLAM(ref_poses, ref_descriptors[args.descriptor], args.wContrast, args.numVel, args.vMin, args.vMax, args.matchWindow, args.enhance)
    proposalSS, scoreSS = seqslam.localize(query_descriptors)
    # setup single image matching
    single = SingleImageMatching(ref_poses, ref_descriptors[args.descriptor])

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
            axs[i].set_xlim(np.min(ref_poses_x) - 100, np.max(ref_poses_x) + 100)
            axs[i].set_ylim(np.min(ref_poses_y) - 100, np.max(ref_poses_y) + 100)

    localizedT = False
    localizedP = False

    for t in range(L):
        if t > 0:
            particlefilter.update(vo[t-1], query_descriptors[t, :])
            topofilter.update(query_descriptors[t, :])
        proposalP, scoreP = particlefilter.localize()
        proposalT, scoreT = topofilter.localize()
        proposalS, scoreS = single.localize_descriptor(query_descriptors[t, :])
        # visualize initial
        if t == 0:
            axs[0].scatter(ref_poses_x, ref_poses_y, c=topofilter.belief, cmap='jet', s=8, vmin=0, vmax=2e-4)
            axs[0].set_title(r"t = {}, $\tau_t$ = {:.2f}".format(t, scoreT), fontsize=24)
            axs[0].scatter([proposalS.t()[1]], [proposalS.t()[0]], color='orange', s=150, marker='*')
            axs[0].scatter([query_poses_x[t]], [query_poses_y[t]], color='lawngreen', s=100, marker='X')
        # visualize intermediate
        if t == args.niter:
            axs[1].scatter(ref_poses_x, ref_poses_y, c=topofilter.belief, cmap='jet', s=8, vmin=0, vmax=2e-4)
            axs[1].set_title(r"t = {}, $\tau_t$ = {:.2f}".format(t, scoreT), fontsize=24)
            axs[1].scatter([proposalS.t()[1]], [proposalS.t()[0]], color='orange', s=150, marker='*')
            axs[1].scatter([query_poses_x[t]], [query_poses_y[t]], color='lawngreen', s=100, marker='X')
        # visualize final
        if scoreT > 0.95 and not localizedT:
            axs[2].scatter(ref_poses_x, ref_poses_y, c=topofilter.belief, cmap='jet', s=8, vmin=0, vmax=1e-2)
            axs[2].set_title(r"t = {}, $\tau_t$ = {:.2f} $\rightarrow$ localize".format(t, scoreT), fontsize=24)
            axs[2].scatter([proposalS.t()[1]], [proposalS.t()[0]], color='orange', s=150, marker='*')
            axs[2].scatter([query_poses_x[t]], [query_poses_y[t]], color='lawngreen', s=100, marker='X')
            # inlet topological
            axs[3].scatter(ref_poses_x, ref_poses_y, c=topofilter.belief, cmap='jet', s=8, vmin=0, vmax=1e-1)
            axs[3].set_xlim(query_poses_x[t] - 10, query_poses_x[t] + 10)
            axs[3].set_ylim(query_poses_y[t] - 5, query_poses_y[t] + 5)
            axs[3].scatter([proposalS.t()[1]], [proposalS.t()[0]], color='orange', s=300, marker='*')
            axs[3].scatter([query_poses_x[t]], [query_poses_y[t]], color='lawngreen', s=200, marker='X')
            axs[3].scatter([proposalT.t()[1]], [proposalT.t()[0]], color='pink', s=300, marker='*')
            # set as localized
            localizedT = True
        # inlet PF
        if scoreP < 3 and not localizedP:
            axs[4].scatter(particlefilter.particles.t()[:, 1], particlefilter.particles.t()[:, 0], cmap='jet', c=particlefilter.weights, s=8)
            axs[4].scatter(ref_poses_x, ref_poses_y, color='black', s=8)
            axs[4].scatter([proposalS.t()[1]], [proposalS.t()[0]], color='orange', s=300, marker='*')
            axs[4].scatter([query_poses_x[t]], [query_poses_y[t]], color='lawngreen', s=200, marker='X')
            axs[4].scatter([proposalP.t()[1]], [proposalP.t()[0]], color='pink', s=300, marker='*')
            axs[4].set_xlim(query_poses_x[t] - 10, query_poses_x[t] + 10)
            axs[4].set_ylim(query_poses_y[t] - 5, query_poses_y[t] + 5)
            # set as localized
            localizedP = True
        
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(utils.figures_path + '/catchy_figure.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate first system figure for paper")
    parser.add_argument('-r', '--reference-traverse', type=str, default='Overcast',
                        help="reference traverse used to build map")
    parser.add_argument('-q', '--query-traverse', type=str, default='Night',
                        help="query traverse to localize against reference map")
    parser.add_argument('-d', '--descriptor', type=str, default='NetVLAD', help='descriptor type to run model using.')
    parser.add_argument('-nl', '--nloc', type=int, default=5, 
        help="which trial to generate figure for")
    parser.add_argument('-ni', '--niter', type=int, default=10, 
        help="which intermediate iteration to check figure")
    # particle filter parameters
    parser.add_argument('-m', '--nparticles', type=int, default=3000, 
        help="number of particles to use during localisation")
    parser.add_argument('-w', '--attitude-weight', type=float, default=20, 
        help="weight for attitude components of pose distance equal to 1 / d for d being rotation angle (rad) equivalent to 1m translation")
    parser.add_argument('-dp', '--delta-particles', type=float, default=10, 
        help="multiple used for calibrating sensor update rate parameter. assumes 6sigma change in image difference causes m times sensor update")
    parser.add_argument('-l2', '--lambda2', type=float, default=0.3, 
        help="rate parameter for computing pose weights")
    parser.add_argument('-kp', '--k-pose', type=int, default=3, 
        help='number of nearest neighbours keyframes for each particle in observation likelihood')
    # topological filter parameters
    parser.add_argument('-dt', '--delta-topo', type=float, default=10, 
        help="multiple used for calibrating sensor update rate parameter. assumes 6sigma change in image difference causes m times sensor update")
    parser.add_argument('-wl', '--window-lower', type=int, default=-2, 
        help="minimum state transition in transition matrix")
    parser.add_argument('-wu', '--window-upper', type=int, default=10, 
        help="maximum state transition in transition matrix")
    # SeqSLAM parameters
    parser.add_argument('-wc', '--wContrast', type=int, default=10, 
        help="window used for local contrast enhancement in difference matrix")
    parser.add_argument('-nv', '--numVel', type=int, default=20, 
        help="number of velocities to search for sequence matching")
    parser.add_argument('-vm', '--vMin', type=float, default=1, 
        help="minimum multiple of query to reference velocity")
    parser.add_argument('-vM', '--vMax', type=float, default=10, 
        help="maximum multiple of query to reference velocity")
    parser.add_argument('-wm', '--matchWindow', type=int, default=20, 
        help="window used for scoring optimal trajectories for each reference")
    parser.add_argument('-e', '--enhance', action='store_true', help='apply contrast enhancement')
    args = parser.parse_args()
    main(args)