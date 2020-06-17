import sys
import os
import argparse
import pickle

import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg

import seaborn as sns
import cv2
from src import geometry

################## filepaths ###################
file_dir = os.path.dirname(os.path.abspath(__file__))
raw_path = os.path.abspath(os.path.join(file_dir, "../../", "data/raw/RobotCar/"))
processed_path = os.path.abspath(os.path.join(file_dir, "../../", "data/processed/RobotCar/"))
results_path = os.path.abspath(os.path.join(file_dir, "../../", "models", "PF_aux.pickle"))
# results_path = os.path.abspath(os.path.join(file_dir, "../../", "models", "PF.pickle"))
# results_path = os.path.abspath(os.path.join(file_dir, "../../", "models", "temp_gt.npz"))
################################################

################## map dimensions ###################
y_min = 619600
y_max = 620690
x_min = 5735130
x_max = 5736120
#####################################################
camStr = "stereo_wide_left" # can use other camera models like mono_*
lut = np.fromfile(os.path.join(raw_path,"camera-models/{}_distortion_lut.bin".format(camStr)), np.double)

def ox_undistort(imgPath,lut):
    imRaw1 = cv2.imread(imgPath,cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)

    if camStr[0]=='s':
        image = cv2.cvtColor(imRaw1, cv2.COLOR_BAYER_GB2BGR)
    else:
        image = cv2.cvtColor(imRaw1, cv2.COLOR_BAYER_RG2BGR)

    lut = lut.reshape([2, lut.size // 2])
    bilinear_lut = lut.transpose()
    
    if image.shape[0] * image.shape[1] != bilinear_lut.shape[0]:
            raise ValueError('Incorrect image size for camera model')

    lut = bilinear_lut[:, 1::-1].T.reshape((2, image.shape[0], image.shape[1]))

    if len(image.shape) == 1:
        raise ValueError('Undistortion function only works with multi-channel images')

    undistorted = np.rollaxis(np.array([map_coordinates(image[:, :, channel], lut, order=1)
                                        for channel in range(0, image.shape[2])]), 0, 3)

    return undistorted.astype(image.dtype)


def main(args):
    w = args.attitude_weight
    R = args.radius
    if args.weight_type == 'weight':
        vmin = 0
        vmax = 1e-3
        score_type = 'weights'
    elif args.weight_type == 'vsim':
        vmin = 0
        vmax = 4
        score_type = 'visual_sims'
    else:
        vmin = 0
        vmax = 1
        score_type = 'pose_wts'
    ################### data ###########################
    loc = args.nloc # which traverse to visualize
    # load generated results from model run
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    proposals = results["proposals"][loc]
    scores = results["scores"][loc]
    particles = results["particles"][loc]
    weights = results[score_type][loc]
    L = len(particles) # max sequence len
    # load query traverse poses and timestamps
    query = results['query']
    query_gt = results['query_gt'][loc]
    query_tstamps = results['query_tstamps'][loc]
    # load reference map
    ref = results['reference']
    ref_gt = results['ref_gt']
    ref_gt_x, ref_gt_y = ref_gt.t()[:, 1], ref_gt.t()[:, 0]
    ref_tstamps = results['ref_tstamps']
    print("Loaded results, processing images...")
    # query traverse images
    query_gt_img_paths = [os.path.join(raw_path, str(query), 'stereo/left', str(query_tstamps[t]) + '.png') for t in range(L)]
    query_gt_imgs = [ox_undistort(img, lut) for img in query_gt_img_paths]
    # closest reference images to query gt
    ref_gt_ind = [np.argmin(geometry.metric(ref_gt, query_gt[t], w)) for t in range(L)]
    ref_gt_img_paths = [os.path.join(raw_path, str(ref), 'stereo/left', str(ref_tstamps[i]) + '.png') for i in ref_gt_ind]
    ref_gt_imgs = [ox_undistort(img, lut) for img in ref_gt_img_paths]
    # closest ref image to highest weight particle
    max_wt_ind = np.argmax(weights, axis=1)
    ref_max_ind = [np.argmin(geometry.metric(ref_gt, particles[t][i], w)) for t, i in enumerate(max_wt_ind)]
    ref_max_img_paths = [os.path.join(raw_path, str(ref), 'stereo/left', str(ref_tstamps[i]) + '.png') for i in ref_max_ind]
    ref_max_imgs = [ox_undistort(img, lut) for img in ref_max_img_paths]
    print("Processsed, generating figures:")
    #######################################################
    #################### setup figures ##################
    fig, axs = plt.subplots(2, 3)
    ref_map_plots = []
    particles_plots = []
    proposal_plots = []
    gt_plots = []
    # initialize map particle plots
    for i, ax in enumerate(axs[0]):
        ax.plot(ref_gt_x, ref_gt_y, color='black') # reference map plots
        ref_map_plots.append(ax.plot([], [], color='black')[0]) # reference traverse map
        particles_plots.append(ax.scatter([], [], c=[], cmap='jet', vmin=vmin, vmax=vmax))
        proposal_plots.append(ax.scatter([], [], color='g', marker="*"))
        gt_plots.append(ax.scatter([], [], color='r', marker="*"))
        ax.set_aspect('equal')
    for i, plot in enumerate(particles_plots): # setup colorbar
        fig.colorbar(plot, ax=axs[0, i])
    # initialize image plots
    img_plots = []
    for i, ax in enumerate(axs[1]):
        ax.set_xticks([])
        ax.set_yticks([])
        img_plots.append(ax.imshow(np.zeros_like(ref_gt_imgs[0])))
    # fixed axes limits for global view
    axs[0, 0].set_xlim(np.min(ref_gt_x) - 100, np.max(ref_gt_x) + 100)
    axs[0, 0].set_ylim(np.min(ref_gt_y) - 100, np.max(ref_gt_y) + 100)
    # update image plot titles
    axs[1, 0].set_title("Query image")
    axs[1, 1].set_title("Closest reference image")
    axs[1, 2].set_title("Highest weight particle")

    def animate(t):
        # plot is zoomed around gt, adj limits
        axs[0, 1].set_xlim(query_gt[t].t()[1] - R, query_gt[t].t()[1] + R)
        axs[0, 1].set_ylim(query_gt[t].t()[0] - R, query_gt[t].t()[0] + R)
        # plot is zoomed around highest wt particle, adj limits
        axs[0, 2].set_xlim(particles[t][max_wt_ind[t]].t()[1] - R, particles[t][max_wt_ind[t]].t()[1] + R)
        axs[0, 2].set_ylim(particles[t][max_wt_ind[t]].t()[0] - R, particles[t][max_wt_ind[t]].t()[0] + R)
        # update title, model info
        # proposal_dist = geometry.metric(proposals[t], query_gt[t], w) # distance from proposal to gt
        rel = proposals[t].inv() * query_gt[t]
        for ax in axs[0]:
            ax.set_title('t: {} score: {:.1f} dist t: {:.1f} dist R: {:.1f}'.format(t, scores[t], np.linalg.norm(rel.t()), rel.R().magnitude() * 180 / np.pi))
        # update particles
        for plot in particles_plots:
            plot.set_offsets(particles[t].t()[:, 1::-1])
            plot.set_array(weights[t])
        for plot in proposal_plots:
            plot.set_offsets([proposals.t()[t, 1::-1]])
        for plot in gt_plots:
            plot.set_offsets([query_gt.t()[t, 1::-1]])
        # image plots
        img_plots[0].set_data(query_gt_imgs[t])
        img_plots[1].set_data(ref_gt_imgs[t])
        img_plots[2].set_data(ref_max_imgs[t])

        print("t", t, np.sqrt(np.diag(np.cov(particles[t].R().as_rotvec(), rowvar=False))))
        # print("t", t, np.sqrt(np.diag(np.cov(particles[t].t(), rowvar=False))))

        return ref_map_plots + particles_plots + proposal_plots + gt_plots + img_plots
    ani = animation.FuncAnimation(fig, animate, np.arange(args.nstart, L), blit=False, interval=args.delay, repeat=True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visualisation script for results")
    parser.add_argument('-nl', '--nloc', type=int, default=0, 
        help="which localisation to visualise")
    parser.add_argument('-ns', '--nstart', type=int, default=0, 
        help="frame number for start of visualization")
    parser.add_argument('-w', '--attitude-weight', type=float, default=11, 
        help="weight for attitude components of pose distance equal to 1 / d for d being rotation angle (rad) equivalent to 1m translation")
    parser.add_argument('-R', '--radius', type=float, default=20, 
        help="size of zoomed in views of map")
    parser.add_argument('-d', '--delay', type=float, default=500, 
        help="delay (ms) between each frame")
    parser.add_argument('-t', '--weight-type', type=str, default='weight', help='type of weight to visualize in colormap')
    args = parser.parse_args()
    weight_type = ['vsim', 'pose', 'weight']
    if args.weight_type not in weight_type:
        parser.error('Invalid weight type for --weight-type. Valid options are: {}'.format(", ".join(weight_type)))
    main(args)
