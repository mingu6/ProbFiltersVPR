import os
import argparse
import pickle
import time

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

from src import geometry, utils, params
from src.thirdparty.nigh import Nigh


class ParticleFilter:
    def __init__(self, poses_tree, map_poses, map_descriptors, n_particles,
                 lambda2, k_pose, delta, w, sigma_init, sigma_vo,
                 auxiliary=False, odom_only=False):
        # model parameters
        self.lambda1 = 0.
        self.lambda2 = lambda2
        self.k_pose = k_pose
        self.delta = delta
        self.w = w
        self.sigma_init = sigma_init
        self.sigma_vo = sigma_vo
        # map keyframe information
        if len(map_poses) != len(map_descriptors):
            raise ValueError("Number of poses and descriptors not the same!")
        self.map_poses = map_poses
        # times 2 scaling is constant offset b/w quat metric and angle
        self.poses_tree = poses_tree
        self.descriptors = map_descriptors
        # particles and weights
        self.n_particles = n_particles
        self.particles = None
        self.weights = None
        # model type
        self.auxiliary = auxiliary
        self.odom_only = odom_only

    def initialize_model(self, query_descriptor):
        dists = np.linalg.norm(self.descriptors -
                               query_descriptor[np.newaxis, :], axis=1)
        # calibrate rate parameter
        descriptor_quantiles = np.quantile(dists, [0.025, 0.975])
        self.lambda1 = np.log(self.delta) / (descriptor_quantiles[1]
                                             - descriptor_quantiles[0])
        # initialize particles
        sims = np.exp(- self.lambda1 * dists)
        map_idx = np.minimum(self._resample_systematic(sims),
                             len(self.map_poses) - 1)  # resample map indices
        particles_noise = np.random.normal(
            scale=self.sigma_init[np.newaxis, :], size=(self.n_particles, 6))
        self.particles = self.map_poses[map_idx] * \
            geometry.expSE3(particles_noise)
        self.weights = np.ones(shape=self.n_particles) / self.n_particles

    def update(self, vo, query_descriptor):
        if self.auxiliary:
            # compute auxiliary lookforward point and weights
            aux_pts = self.apply_motion(vo, noise=False)
            aux_wts = self.apply_measurement(aux_pts, query_descriptor)
            # resample particles using auxiliary weights and apply motion
            self.resample_particles(aux_wts * self.weights)
            self.particles = self.apply_motion(vo, noise=True)
            # adjust weights based on current particles weights
            wt_update = self.apply_measurement(self.particles,
                                               query_descriptor)
            self._weight_update(wt_update / aux_wts)
        else:
            self.particles = self.apply_motion(vo, noise=True)
            update = self.apply_measurement(self.particles, query_descriptor)
            self._weight_update(update)
            self.resample_particles(self.weights)
        return None

    def apply_motion(self, vo, noise=True):
        sigma = self.sigma_vo
        # add noise to raw vo output if required
        vo_noisy = vo
        if noise:
            vo_noise = np.random.normal(scale=sigma,
                                        size=(self.n_particles, 6))
            vo_noisy = geometry.expSE3(vo_noise) * vo
        # apply noisy motion
        return self.particles * vo_noisy

    def apply_measurement(self, particles, query_descriptor):
        # identify nearest reference keyframe to particle sets
        gt_dist, gt_ind = self.poses_tree.nearest(
            particles.t(), particles.R().as_quat(), self.k_pose, 4)
        # evaluate descriptor distances on unique pairs only
        u_gt_ind, inv_ind = \
            np.unique(gt_ind, return_inverse=True)
        pose_wts = self._pose_wts(gt_dist)  # pose weight (phi)
        if not self.odom_only:
            # for some reason np.dot slows down
            # nearest neighbor search significantly
            desc_dist = np.linalg.norm(
                self.descriptors[u_gt_ind] -
                query_descriptor[np.newaxis, :],
                axis=1)
            # visual similarity (alpha)
            visual_sim = self._visual_sim(desc_dist)
            update = np.sum(pose_wts *
                            visual_sim[inv_ind].reshape(gt_dist.shape), axis=1)
        else:
            update = np.sum(pose_wts, axis=1)
        return update

    def resample_particles(self, weights):
        if self.auxiliary:
            resample_idx = self._resample_systematic(weights)
            self.particles = self.particles[resample_idx]
            self.weights = np.ones(self.n_particles) / self.n_particles
        else:
            ESS = np.sum(weights ** 2) ** -1
            if ESS < self.n_particles * 0.5:
                resample_idx = self._resample_systematic(weights)
                self.particles = self.particles[resample_idx]
                self.weights = np.ones(self.n_particles) / self.n_particles

    def localize(self):
        # weighted mean of particles is the proposal
        max_wt_particle = self.particles[np.argmax(self.weights)]
        dists = geometry.metric(max_wt_particle, self.particles, self.w)
        cluster_idx = dists < 10
        confidence = np.sum(self.weights[cluster_idx])
        t_avg = np.average(
            np.atleast_2d(self.particles[cluster_idx].t()),
            axis=0, weights=self.weights[cluster_idx])
        R_avg = self.particles[cluster_idx].R().mean(self.weights[cluster_idx])
        return geometry.SE3Poses(t_avg, R_avg), confidence

    def _weight_update(self, update):
        self.weights *= update
        self.weights /= self.weights.sum()

    def _resample_systematic(self, weights):
        u = (np.random.uniform(0, 1) +
             np.arange(0, self.n_particles, 1)) / self.n_particles
        return np.searchsorted(
            np.cumsum(weights / weights.sum()), u, side='left')

    def _pose_wts(self, dist):
        return np.exp(- dist * self.lambda2)
        # return np.exp(- np.maximum(dist-3, 0) * self.lambda2)

    def _visual_sim(self, descriptor_dists):
        log_sim = - descriptor_dists * self.lambda1
        # subtracting median allows for numerical stability
        return np.exp(log_sim - np.median(log_sim))


def main(args):
    # load reference data
    ref_poses, ref_descriptors, _ = \
        utils.import_reference_map(args.reference_traverse)
    # NN search tree for poses
    # 2 times is b/c of rotation angle representation in library
    ref_tree = Nigh.SE3Tree(2 * args.attitude_weight)
    ref_tree.insert(ref_poses.t(), ref_poses.R().as_quat())
    # localize all selected query traverses
    pbar = tqdm(args.query_traverses)
    for traverse in pbar:
        pbar.set_description(traverse)
        # savepath
        save_path = os.path.join(utils.results_path, traverse)
        # load query data
        query_poses, vo, rtk_motion, query_descriptors, _ =\
            utils.import_query_traverse(traverse)
        # regular traverse with VO
        pbar = tqdm(args.descriptors, leave=False)
        for desc in pbar:
            pbar.set_description(desc)
            save_path1 = os.path.join(
                save_path, desc)  # one folder per descriptor
            if not os.path.exists(save_path1):
                os.makedirs(save_path1)
            # save results from all trials
            proposals_all, scores_all, times_all = [], [], []
            for _ in trange(args.ntrials, leave=False, desc='Trials'):
                model = ParticleFilter(ref_tree, ref_poses,
                                       ref_descriptors[desc], args.nparticles,
                                       args.lambda2, args.k_pose,
                                       args.delta, args.attitude_weight,
                                       params.sigma_init,
                                       params.sigma_vo[traverse])
                proposals, scores, times =\
                    utils.localize_traverses_filter(
                        model, query_descriptors[desc], gt=query_poses,
                        vo=vo, desc='Regular VO')
                proposals_all.append(proposals)
                scores_all.append(scores)
                times_all.append(times)
            utils.save_obj(
                save_path1 + '/MCL.pickle', model='MCL',
                query_gt=query_poses, proposals=proposals_all,
                scores=scores_all, times=times_all)
        # RTK motion ablation
        if not args.regular_only:
            pbar = tqdm(args.descriptors, leave=False)
            for desc in pbar:
                pbar.set_description(desc)
                save_path1 = os.path.join(save_path, desc)
                proposals_all, scores_all, times_all = [], [], []
                for _ in trange(args.ntrials, leave=False, desc='Trials'):
                    model = ParticleFilter(
                        ref_tree, ref_poses, ref_descriptors[desc],
                        args.nparticles, args.lambda2, args.k_pose,
                        args.delta, args.attitude_weight, params.sigma_init,
                        params.sigma_vo[traverse])
                    proposals, scores, times =\
                        utils.localize_traverses_filter(
                            model, query_descriptors[desc], vo=rtk_motion,
                            desc='RTK motion')
                    proposals_all.append(proposals)
                    scores_all.append(scores)
                    times_all.append(times)
                utils.save_obj(
                    save_path1 + '/MCL_RTK_motion.pickle',
                    model='MCL RTK motion', query_gt=query_poses,
                    proposals=proposals_all, scores=scores_all,
                    times=times_all)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCL algorithm on trials")
    parser.add_argument('-r', '--reference-traverse',
                        type=str, default='Overcast',
                        help="reference traverse used as the map")
    parser.add_argument('-q', '--query-traverses', nargs='+', type=str,
                        default=['Sun', 'Dusk', 'Night'],
                        help="Names of query traverses to localize against"
                        "reference map e.g. Overcast, Night, Dusk etc."
                        "Input 'all' instead to process all traverses."
                        "See src/params.py for full list.")
    parser.add_argument('-d', '--descriptors', nargs='+', type=str,
                        default=['NetVLAD', 'DenseVLAD'],
                        help='descriptor types to run experiments on.')
    parser.add_argument('-N', '--ntrials', type=int, default=5,
                        help="number of trials to run")
    parser.add_argument('-M', '--nparticles', type=int, default=6000,
                        help="number of particles to use during localisation")
    parser.add_argument('-W', '--attitude-weight', type=float, default=15,
                        help="weight for attitude components of pose"
                        "distance equal to 1 / d for d being rotation angle"
                        "(rad) equivalent to 1m translation")
    parser.add_argument('-D', '--delta', type=float, default=5,
                        help="multiple used for calibrating sensor update"
                        "rate parameter. assumes 6sigma change in image"
                        "difference causes m times sensor update")
    parser.add_argument('-l2', '--lambda2', type=float, default=0.2,
                        help="rate parameter for computing pose weights")
    parser.add_argument('-kp', '--k-pose', type=int, default=3,
                        help="number of nearest neighbours keyframes for"
                        "each particle in observation likelihood")
    parser.add_argument('-a', '--auxiliary', action='store_true',
                        help="use auxiliary particle filter instead"
                        "of bootstrap")
    parser.add_argument('-g', '--gt-motion', action='store_true',
                        help='use relative motion from RTK instead of VO')
    parser.add_argument('-o', '--odom-only', action='store_true',
                        help='use odometry component only')
    parser.add_argument('-re', '--regular-only', action='store_true',
                        help='regular vo only')
    args = parser.parse_args()
    main(args)
