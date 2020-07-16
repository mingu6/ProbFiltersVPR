import os
import argparse
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from matplotlib.colors import Normalize

from src import geometry, utils, params
from src.settings import READY_PATH

from src.models.ParticleFilter import ParticleFilter
from src.models.TopologicalFilter import TopologicalFilter
from src.models.SeqMatching import SeqMatching
from src.models.SingleImageMatching import SingleImageMatching
from src.thirdparty.nigh import Nigh

# ranges for scores in visualization
vmins = [0., 0.] # PF, Topo
vmaxs = [1e-3, 2e-1] # PF, Topo

def main(args):
    # import reference and query traverses
    ref_poses, ref_descriptors, ref_tstamps = utils.import_reference_map(args.reference_traverse)
    query_poses, vo, _, query_descriptors, query_tstamps = utils.import_query_traverse(args.query_traverse)
    # shortcuts used when plotting
    ref_poses_x, ref_poses_y = ref_poses.t()[:, 1], ref_poses.t()[:, 0]
    query_descriptors = query_descriptors[args.descriptor][args.nloc]
    query_poses = query_poses[args.nloc]
    query_poses_x, query_poses_y = query_poses.t()[:, 1], query_poses.t()[:, 0]
    query_tstamps = query_tstamps[args.nloc]
    L = len(vo[args.nloc]) + 1 # max sequence length
    vo = vo[args.nloc]
    # import all images
    query_images = []
    nearest_ref_images = []
    for t in range(len(query_tstamps)):
        # load query images
        imgFolderQ = os.path.join(READY_PATH, params.traverses[args.query_traverse], 'stereo/left')
        imgPathQ = os.path.join(imgFolderQ, str(query_tstamps[t]) + '.png')
        imgQ = plt.imread(imgPathQ)
        query_images.append(imgQ)

        distrel = geometry.metric(query_poses[t], ref_poses, args.attitude_weight)
        idx = np.argmin(distrel)
        imgFolderR = os.path.join(READY_PATH, params.traverses[args.reference_traverse], 'stereo/left')
        imgPathR = os.path.join(imgFolderR, str(ref_tstamps[idx]) + '.png')
        imgR = plt.imread(imgPathR)
        nearest_ref_images.append(imgR)
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
    seqmatch = SeqMatching(ref_poses, ref_descriptors[args.descriptor], args.wContrast, args.numVel, args.vMin, args.vMax, args.matchWindow, args.enhance)
    proposalSS, scoreSS = seqmatch.localize(query_descriptors)
    # setup single image matching
    single = SingleImageMatching(ref_poses, ref_descriptors[args.descriptor])

    # initialize over figure and subplots
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    # initialize specific particle/belief plots
    belief_plots = []
    proposal_plots = []
    true_query_plots = []
    single_plots = []
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1] - 1):
            if (i == 0 and j == 0) or (i == 1 and j == 0):
                axs[i, j].scatter(ref_poses_x, ref_poses_y, color='black', s=8) # reference map plots (Topo)
                belief_plots.append(axs[i, j].scatter([], [], cmap='jet', c=[], vmin=vmins[j], vmax=vmaxs[j]))
            else:
                belief_plots.append(axs[i, j].scatter(ref_poses_x, ref_poses_y, s=8, cmap='jet', c=topofilter.belief, vmin=vmins[j], vmax=vmaxs[j]))
            proposal_plots.append(axs[i, j].scatter([], [], color='gold', marker="*", s=75))
            true_query_plots.append(axs[i, j].scatter([], [], color='lawngreen', marker="X", s=50))
            single_plots.append(axs[i, j].scatter([], [], color='orange', marker="*", s=50))
            # plot seqslam proposals once
            axs[i, j].scatter([proposalSS.t()[1]], [proposalSS.t()[0]], color='magenta', marker="*", s=50)
            axs[i, j].scatter([query_poses_x[0]], [query_poses_y[0]], color='magenta', marker="X", s=50)
            # fixed axes limits for global view
            axs[i, j].set_aspect('equal', adjustable='datalim')
            if (i == 0 and j == 1) or (i == 1 and j == 1):
                axs[i, j].set_xlim(np.min(ref_poses_x) - 100, np.max(ref_poses_x) + 100)
                axs[i, j].set_ylim(np.min(ref_poses_y) - 100, np.max(ref_poses_y) + 100)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    # image plots
    axs[0, -1].set_xticks([])
    axs[0, -1].set_yticks([])
    axs[1, -1].set_xticks([])
    axs[1, -1].set_yticks([])
    img_plots = [axs[0, -1].imshow(np.zeros_like(query_images[0])), axs[1, -1].imshow(np.zeros_like(query_images[0]))]
    axs[0, 2].set_title("Query image")
    axs[1, 2].set_title("Closest reference image")
        
    def animate(t):
        R = args.radius
        # update belief
        if t > 0:
            particlefilter.update(vo[t-1], query_descriptors[t, :])
            topofilter.update(query_descriptors[t, :])
        proposalP, scoreP = particlefilter.localize()
        proposalT, scoreT = topofilter.localize()
        proposalS, scoreS = single.localize_descriptor(query_descriptors[t, :])
        # plot beliefs for all models
        particles = particlefilter.particles
        belief = topofilter.belief
        belief_plots[0].set_offsets(particles.t()[:, 1::-1]) # particle filter plots
        belief_plots[2].set_offsets(particles.t()[:, 1::-1])
        belief_plots[0].set_array(particlefilter.weights)
        belief_plots[2].set_array(particlefilter.weights)
        belief_plots[1].set_array(belief)
        belief_plots[3].set_array(belief)
        # plot is zoomed around gt, adj limits
        for j in range(axs.shape[1] - 1):
            axs[1, j].set_xlim(query_poses_x[t] - R, query_poses_x[t] + R)
            axs[1, j].set_ylim(query_poses_y[t] - R, query_poses_y[t] + R)
        
        # compute proposal distance from truth
        relP = proposalP.inv() * query_poses[t]
        relT = proposalT.inv() * query_poses[t]
        tdistP = np.linalg.norm(relP.t())
        tdistT = np.linalg.norm(relT.t())
        RdistP = relP.R().magnitude() * 180 / np.pi
        RdistT = relT.R().magnitude() * 180 / np.pi
        axs[0, 0].set_title(r"MCL: $\tau_t = {:.1f}$, t: {:.1f} R: {:.1f}".format(scoreP, tdistP, RdistP))
        axs[1, 0].set_title(r"MCL: Local view. $\tau_t = {:.1f}$, t: {:.1f} R: {:.1f}".format(scoreP, tdistP, RdistP))
        axs[0, 1].set_title(r"Topological: Global view. $\tau_t = {:.2f}$, t: {:.1f} R: {:.1f}".format(scoreT, tdistT, RdistT))
        axs[1, 1].set_title(r"Topological: Local view. $\tau_t = {:.2f}$, t: {:.1f} R: {:.1f}".format(scoreT, tdistT, RdistT))
        # plot proposals from each model
        proposal_plots[0].set_offsets([proposalP.t()[1::-1]]) # particles
        proposal_plots[2].set_offsets([proposalP.t()[1::-1]])
        proposal_plots[1].set_offsets([proposalT.t()[1::-1]]) # topological
        proposal_plots[3].set_offsets([proposalT.t()[1::-1]])
        single_plots[0].set_offsets([proposalS.t()[1::-1]]) # particles
        single_plots[2].set_offsets([proposalS.t()[1::-1]])
        single_plots[1].set_offsets([proposalS.t()[1::-1]]) # topological
        single_plots[3].set_offsets([proposalS.t()[1::-1]])
        # plot ground truth pose for each panel
        for plot in true_query_plots:
            plot.set_offsets([query_poses_x[t], query_poses_y[t]])
        # image plots
        img_plots[0].set_data(query_images[t])
        img_plots[1].set_data(nearest_ref_images[t])

        return belief_plots + proposal_plots + true_query_plots

    fig.tight_layout()
    ani = animation.FuncAnimation(fig, animate, np.arange(L), blit=False, interval=args.delay, repeat=True)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(utils.figures_path + '/visualization.mp4', writer=writer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate first system figure for paper")
    parser.add_argument('-r', '--reference-traverse', type=str, default='Overcast',
                        help="reference traverse used to build map")
    parser.add_argument('-q', '--query-traverse', type=str, default='Night',
                        help="query traverse to localize against reference map")
    parser.add_argument('-d', '--descriptor', type=str, default='NetVLAD', help='descriptor type to run model using.')
    parser.add_argument('-nl', '--nloc', type=int, default=5, 
        help="which trial to generate figure for")
    parser.add_argument('-R', '--radius', type=float, default=20, 
        help="size of zoomed in views of map")
    parser.add_argument('-de', '--delay', type=float, default=500, 
        help="delay (ms) between each frame")
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
    # Sequence matching parameters
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
