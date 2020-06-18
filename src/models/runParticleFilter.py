import os
import argparse
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

import params
from ParticleFilter import ParticleFilter
from src import geometry, utils
import src
from src.thirdparty.nigh import Nigh

def localize_traverses(reference, query, desc, nparticles, lambda2, kp, delta, w, 
        aux=False, odom_only=False, gt_motion=False):
    # import reference and query traverses
    ref_gt, ref_descriptors, _, query_gt, vo, query_descriptors, _, _, _\
        = utils.import_traverses(reference, query, desc, gt_motion=gt_motion)

    # saved traverse statistics
    proposals = []
    scores = []
    times = []

    # NN search tree for poses
    poses_tree = Nigh.SE3Tree(2 * w)
    poses_tree.insert(ref_gt.t(), ref_gt.R().as_quat())

    nloc = len(vo) # number of localizations
    L = len(vo[0]) + 1 # max sequence length
    # show process bar
    if odom_only:
        description = 'PF Odom only'
    if gt_motion:
        description = 'PF RTK'
    else:
        description = 'PF Regular'
    # localize trials!
    for i in trange(nloc, desc=description, leave=False):
        # save outputs of model
        proposals_seq = []
        score_seq = np.empty(L)
        times_seq = np.empty(L)

        # begin timing localization
        start = time.time()

        # initialize model, particles and calibrate parameters
        model = ParticleFilter(query, nparticles, lambda2, kp, delta, w,
                    poses_tree, ref_gt, ref_descriptors, query_descriptors[i, 0, :], auxiliary=aux, odom_only=odom_only)

        if args.verbose:
            localized = False

        for t in range(L):
            if t > 0:
                model.update(vo[i][t-1], query_descriptors[i, t, :])
            proposal, score = model.localize()
            # save particle and proposal info for evaluation/visualization for sequence
            proposals_seq.append(proposal)
            score_seq[t] = score
            times_seq[t] = time.time() - start
            # check results
            if args.verbose:
                rel = proposal.inv() * query_gt[i][t]
                if np.sum(score) < 4.0 and not localized:
                    print("t", t, "dist t", np.linalg.norm(rel.t()), "dist R", rel.R().magnitude() * 180 / 3.1415, "scores: ", score)
                    localized = True
        proposals_seq = geometry.combine(proposals_seq)
        # save traverse statistics
        proposals.append(proposals_seq)
        scores.append(score_seq)
        times.append(times_seq)
    return proposals, scores, times, query_gt

def save_results(reference, query, proposals, scores, times, query_gt, 
            auxiliary=False, gt_motion=False, odom_only=False):
    # save model output
    save_path_query = os.path.join(utils.save_path, query)
    if not os.path.exists(save_path_query):
        os.makedirs(save_path_query)
    fname = "/PF"
    if auxiliary:
        fname += "_aux"
        model_name = "Auxiliary"
    else:
        model_name = 'Particle Filter'
    if gt_motion:
        savefile = save_path_query + fname + '_RTK.pickle'
        model_name += " RTK motion"
    else:
        savefile = save_path_query + fname + '.pickle'
    if odom_only:
        fname = "/PF_Odom_only"
        savefile = save_path_query + fname + '.pickle'
        model_name = 'Odometry Only'
    utils.save_obj(savefile, model=model_name, reference=reference, query=query, query_gt=query_gt, 
                                    proposals=proposals, scores=scores, times=times)

def main(args):
    if not args.All:
        proposals, scores, times, query_gt = localize_traverses(args.reference_traverse, args.query_traverse, args.descriptor, 
                                args.nparticles, args.lambda2, args.k_pose, args.delta, args.attitude_weight,
                                aux=args.auxiliary, odom_only=args.odom_only, gt_motion=args.gt_motion)
        save_results(args.reference_traverse, args.query_traverse, proposals, scores, times, query_gt)
    else:
        traverses = ['Rain', 'Night', 'Dusk']
        pbar = tqdm(traverses)
        for traverse in pbar:
            pbar.set_description(traverse)
            # regular traverse with VO
            proposals, scores, times, query_gt = localize_traverses(args.reference_traverse, traverse, args.descriptor, 
                                    args.nparticles, args.lambda2, args.k_pose, args.delta, args.attitude_weight,
                                    aux=args.auxiliary, odom_only=False, gt_motion=False)
            save_results(args.reference_traverse, traverse, proposals, scores, times, query_gt)
            # RTK motion ablation
            proposals, scores, times, query_gt = localize_traverses(args.reference_traverse, traverse, args.descriptor, 
                                    args.nparticles, args.lambda2, args.k_pose, args.delta, args.attitude_weight,
                                    aux=args.auxiliary, odom_only=False, gt_motion=True)
            save_results(args.reference_traverse, traverse, proposals, scores, times, query_gt, gt_motion=True)
            # Odometry only
            proposals, scores, times, query_gt = localize_traverses(args.reference_traverse, traverse, args.descriptor, 
                                    args.nparticles, args.lambda2, args.k_pose, args.delta, args.attitude_weight,
                                    aux=args.auxiliary, odom_only=True, gt_motion=True)
            save_results(args.reference_traverse, traverse, proposals, scores, times, query_gt, odom_only=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCL algorithm on trials")
    parser.add_argument('-r', '--reference-traverse', type=str, default='2015-03-17-11-08-44',
                        help="reference traverse used to build map")
    parser.add_argument('-q', '--query-traverse', type=str, default='Night', choices=['Rain', 'Dusk', 'Night'],
                        help="query traverse to localize against reference map")
    parser.add_argument('-M', '--nparticles', type=int, default=5000, 
        help="number of particles to use during localisation")
    parser.add_argument('-W', '--attitude-weight', type=float, default=20, 
        help="weight for attitude components of pose distance equal to 1 / d for d being rotation angle (rad) equivalent to 1m translation")
    parser.add_argument('-D', '--delta', type=float, default=10, 
        help="multiple used for calibrating sensor update rate parameter. assumes 6sigma change in image difference causes m times sensor update")
    parser.add_argument('-l2', '--lambda2', type=float, default=0.3, 
        help="rate parameter for computing pose weights")
    parser.add_argument('-kp', '--k-pose', type=int, default=3, 
        help='number of nearest neighbours keyframes for each particle in observation likelihood')
    parser.add_argument('-a', '--auxiliary', action='store_true', help='use auxiliary particle filter instead of bootstrap')
    parser.add_argument('-g', '--gt-motion', action='store_true', help='use relative motion from RTK instead of VO')
    parser.add_argument('-o', '--odom-only', action='store_true', help='use odometry component only')
    parser.add_argument('--descriptor', type=str, default='NetVLAD', choices=['NetVLAD', 'DenseVLAD'], help='type of descriptor to use. options: NetVLAD')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
    parser.add_argument('-A', '--All', action='store_true', help='run all configurations required to generate results')
    args = parser.parse_args()

    main(args)