import numpy as np
import os
import argparse
import pickle
import time

from tqdm import trange, tqdm
import matplotlib.pyplot as plt

from src import geometry, utils

class TopologicalFilter:
    def __init__(self, delta, map_poses, map_descriptors, query_descriptors_0,
                    window_lower=-2, window_upper=10):
        # store map data
        self.map_poses = map_poses
        self.map_descriptors = map_descriptors
        # initialize hidden states and obs lhood parameters
        self.delta = delta
        self.lambda1 = 0.0
        self.belief = None
        self.init_parameters(query_descriptors_0)
        # parameters for prior transition matrix
        self.window_lower = window_lower
        self.window_upper = window_upper
        self.window_size = int((window_upper - window_lower) / 2)
        self.transition = np.ones(window_upper - window_lower)

    def init_parameters(self, descriptor):
        dists = np.linalg.norm(self.map_descriptors - descriptor[np.newaxis, :], axis=1)
        descriptor_quantiles = np.quantile(dists, [0.025, 0.975])
        self.lambda1 = np.log(self.delta) / (descriptor_quantiles[1] - descriptor_quantiles[0])
        self.belief = np.exp(-self.lambda1 * dists)
        self.belief /= self.belief.sum()
    
    def obs_lhood(self, descriptor):
        vsim = np.exp(-self.lambda1 * np.linalg.norm(descriptor[np.newaxis, :] - self.map_descriptors, axis=1))
        return vsim

    def update(self, descriptor):
        w_l = self.window_lower
        if w_l < 0:
            conv_ind_l, conv_ind_h = np.abs(w_l), len(self.belief) + np.abs(w_l)
            bel_ind_l, bel_ind_h = 0, len(self.belief)
        else:
            conv_ind_l, conv_ind_h = 0, len(self.belief) - w_l
            bel_ind_l, bel_ind_h = w_l, len(self.belief)
        # apply prior transition matrix
        self.belief[bel_ind_l:bel_ind_h] = np.convolve(self.belief, self.transition)[conv_ind_l:conv_ind_h]
        if w_l > 0:
            self.belief[:w_l] = 0.0
        self.belief *= self.obs_lhood(descriptor) # observation likelihood update
        self.belief /= self.belief.sum()

    def localize(self):
        max_bel = np.argmax(self.belief)
        nhood_inds = np.arange(max(max_bel - 2 * self.window_size, 0), min(max_bel + 2 * self.window_size, len(self.belief) - 1))
        score = np.sum(self.belief[nhood_inds]) 
        proposal = int(np.rint(np.average(nhood_inds, weights=self.belief[nhood_inds])))
        return proposal, score

def localize_traverses(reference, query, desc, delta, w_l, w_u, verbose=False):
    ref_gt, ref_descriptors, _, query_gt, _, query_descriptors, _, _, _\
        = utils.import_traverses(reference, query, desc)
    # experiment parameters
    nloc = len(query_gt)
    L = len(query_gt[0])
    # store model output
    proposals = []
    scores = []
    times = []
    for i in trange(nloc, desc='Topological', leave=False):
        scores_seq = np.zeros(L)
        proposals_seq = []
        times_seq = np.zeros(L)

        start = time.time() # start timing
        model = TopologicalFilter(args.delta, ref_gt, ref_descriptors, query_descriptors[i, 0, :], 
                window_lower=args.window_lower, window_upper=args.window_upper)
        if verbose:
            localized = False
        for t in range(L):
            if t > 0:
                model.update(query_descriptors[i, t, :])
            # save output for iteration
            proposal, score = model.localize()
            proposals_seq.append(proposal)
            scores_seq[t] = score
            times_seq[t] = time.time() - start
            if verbose:
                rel = ref_gt[int(proposal)].inv() * query_gt[i][t]
                if score > 0.95 and not localized:
                    print("loc: ", i, "t: ", t, "t dist:", np.linalg.norm(rel.t()), 'R dist', rel.R().magnitude() * 180 / 3.1415, "score:", score)
                    localized = True
        proposals.append(ref_gt[np.asarray(proposals_seq, dtype=int)])
        scores.append(scores_seq)
        times.append(times_seq)
    return proposals, scores, times, query_gt

def save_results(reference, query, proposals, scores, times, query_gt):
    save_path_query = os.path.join(utils.save_path, query)
    if not os.path.exists(save_path_query):
        os.makedirs(save_path_query)
    savefile = save_path_query + '/Topological.pickle'
    name = "Topological Filter"
    utils.save_obj(savefile, model=name, reference=reference, query=query, query_gt=query_gt, 
                                   proposals=proposals, scores=scores, times=times)

def main(args):
    if not args.All:
        proposals, scores, times, query_gt = localize_traverses(args.reference_traverse, args.query_traverse, args.descriptor, 
                            args.delta, args.window_lower, args.window_upper, verbose=args.verbose)
        save_results(args.reference_traverse, args.query_traverse, proposals, scores, times, query_gt)
    else:
        traverses = ['Rain', 'Night', 'Dusk']
        pbar = tqdm(traverses)
        for traverse in pbar:
            pbar.set_description(traverse)
            proposals, scores, times, query_gt = localize_traverses(args.reference_traverse, traverse, args.descriptor, 
                                args.delta, args.window_lower, args.window_upper)
            save_results(args.reference_traverse, traverse, proposals, scores, times, query_gt)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run topological filter trials")
    parser.add_argument('-r', '--reference-traverse', type=str, default='2015-03-17-11-08-44',
                        help="reference traverse used to build map")
    parser.add_argument('-q', '--query-traverse', type=str, default='Night', choices=['Rain', 'Dusk', 'Night'],
                        help="query traverse to localize against reference map")
    parser.add_argument('-D', '--delta', type=float, default=10, 
        help="multiple used for calibrating sensor update rate parameter. Assumes 6sigma change in image difference causes M times sensor update")
    parser.add_argument('-wl', '--window-lower', type=int, default=-2, 
        help="minimum state transition in transition matrix")
    parser.add_argument('-wu', '--window-upper', type=int, default=10, 
        help="maximum state transition in transition matrix")
    parser.add_argument('--descriptor', type=str, default='NetVLAD', choices=['NetVLAD', 'DenseVLAD'], help='type of descriptor to use. options: NetVLAD')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
    parser.add_argument('-A', '--All', action='store_true', help='run all configurations required to generate results')
    args = parser.parse_args()

    main(args)