import numpy as np

from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

from src import geometry
import params

deg = np.pi / 180

class ParticleFilter:
    def __init__(self, traverse, n_particles, lambda2, k_pose, delta, w, poses_tree, map_poses, map_descriptors, query_descriptor_0,
                    auxiliary=False, odom_only=False):
        # model parameters
        self.traverse = traverse
        self.lambda1 = 0.
        self.lambda2 = lambda2
        # self.pose_bw = 4
        self.k_pose = k_pose
        self.delta = delta
        self.w = w
        # map keyframe information
        if len(map_poses) != len(map_descriptors):
            raise ValueError("Number of poses and descriptors not the same!")
        self.map_poses = map_poses
        self.poses_tree = poses_tree # times 2 scaling is constant offset b/w quat metric and angle
        self.descriptors = map_descriptors
        # particles and weights
        self.n_particles = n_particles
        self.particles = None
        self.weights = None
        # model type
        self.auxiliary = auxiliary
        self.odom_only = odom_only
        # initialize model
        self.initialize_model(query_descriptor_0)

    def initialize_model(self, query_descriptor):
        dists = np.linalg.norm(query_descriptor[np.newaxis, :] - self.descriptors, axis=1)
        # calibrate rate parameter
        descriptor_quantiles = np.quantile(dists, [0.025, 0.975])
        self.lambda1 = np.log(self.delta) / (descriptor_quantiles[1] - descriptor_quantiles[0])
        # initialize particles 
        sims = np.exp(- self.lambda1 * dists)
        map_idx = np.minimum(self._resample_stratified(sims), len(self.map_poses) - 1) # resample map indices
        particles_noise = np.random.normal(scale=params.sigma_init[np.newaxis, :], size=(self.n_particles, 6)) 
        self.particles = self.map_poses[map_idx] * geometry.expSE3(particles_noise)
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
            wt_update = self.apply_measurement(self.particles, query_descriptor)
            self._weight_update(wt_update / aux_wts)
        else:
            self.particles = self.apply_motion(vo, noise=True)
            update = self.apply_measurement(self.particles, query_descriptor)
            self._weight_update(update)
            self.resample_particles(self.weights)
        return None

    def apply_motion(self, vo, noise=True):
        ##### fixed parameters #####
        sigma = params.sigma_vo[self.traverse]
        ############################
        # add noise to raw vo output if required
        vo_noisy = vo
        if noise:
            vo_noise = np.random.normal(scale=sigma, size=(self.n_particles, 6))
            vo_noisy = geometry.expSE3(vo_noise) * vo
        # apply noisy motion
        return self.particles * vo_noisy

    def apply_measurement(self, particles, query_descriptor):
        # identify nearest reference keyframe to particle sets
        gt_dist, gt_ind = self.poses_tree.nearest(particles.t(), particles.R().as_quat(), self.k_pose, 1)
        u_gt_ind, inv_ind = np.unique(gt_ind, return_inverse=True) # evaluate descriptor distances on unique pairs only
        pose_wts = self._pose_wts(gt_dist) # pose weight (phi)
        if not self.odom_only:
            desc_dist = np.linalg.norm(self.descriptors[u_gt_ind] - query_descriptor[np.newaxis, :], axis=1)
            visual_sim = self._visual_sim(desc_dist) # visual similarity (alpha)
            # combine using equation *ref*
            update = np.sum(pose_wts * visual_sim[inv_ind].reshape(gt_dist.shape), axis=1)
        else:
            update = np.sum(pose_wts, axis=1)
        return update

    def resample_particles(self, weights):
        if self.auxiliary:
            resample_idx = self._resample_stratified(weights)
            self.particles = self.particles[resample_idx]
            self.weights = np.ones(self.n_particles) / self.n_particles
        else:
            ESS = np.sum(weights ** 2) ** -1
            if ESS < self.n_particles * 0.5:
                resample_idx = self._resample_stratified(weights)
                self.particles = self.particles[resample_idx]
                self.weights = np.ones(self.n_particles) / self.n_particles

    def localize(self):
        # weighted mean of particles is the proposal
        t_avg = np.average(self.particles.t(), axis=0, weights=self.weights)
        R_avg = self.particles.R().mean(self.weights)
        # weighted std dev is the score
        std_t = np.sqrt(np.sum(np.expand_dims(self.weights, 1) * (self.particles.t() - np.expand_dims(t_avg, 0)) ** 2, axis=0))
        std_R = np.sqrt(np.sum(np.expand_dims(self.weights, 1) * (self.particles.R().as_rotvec() - np.expand_dims(R_avg.as_rotvec(), 0)) ** 2, axis=0))
        return geometry.SE3Poses(t_avg, R_avg), np.max(np.concatenate((std_t, self.w * std_R)))
    
    def _weight_update(self, update):
        self.weights *= update
        self.weights /= self.weights.sum()

    def _resample_stratified(self, weights):
        u = (np.random.uniform(0, 1) + np.arange(0, self.n_particles, 1)) / self.n_particles
        return np.searchsorted(np.cumsum(weights / weights.sum()), u, side='left')

    def _pose_wts(self, dist):
        return np.exp(- dist * self.lambda2)

    def _visual_sim(self, descriptor_dists):
        log_sim = - descriptor_dists * self.lambda1
        return np.exp(log_sim - np.median(log_sim)) # subtracting median allows for numerical stability