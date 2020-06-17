import os
import argparse
import pickle
import time

import numpy as np
from tqdm import trange, tqdm

from src import utils
from src import geometry

################## filepaths ###################
file_dir = os.path.dirname(os.path.abspath(__file__))
raw_path = os.path.abspath(os.path.join(file_dir, "../../", "data/raw/RobotCar/"))
processed_path = os.path.abspath(os.path.join(file_dir, "../../", "data/processed/RobotCar/"))
save_path = os.path.abspath(os.path.join(file_dir, "../../", "models"))
################################################

def contrast_enhancement(D, Rwindow):
    """
    Performs contrast enhancement of NxL difference matrix, where N
    is the number of templates and L is the length for a single query
    sequence.

    Args:
        D (NxL numpy array): contains image difference scores
        Rwindow (int > 0): window size around template to apply enhancement
    Returns:
        Denhanced (NxL numpy array): contains contrast enhanced differences
    """
    nref = len(D)
    Denhanced = np.empty_like(D)
    for i in range(nref):
        # reference indices of window around each reference image
        idx_lower = max(i - int(Rwindow / 2), 0)
        idx_upper = min(i + int(Rwindow / 2) + 1, nref - 1)
        # local normalization of window given by indices above
        Denhanced[i, :] = (D[i, :] - np.mean(D[idx_lower:idx_upper, :], axis=0)) / np.std(D[idx_lower:idx_upper, :], axis=0)
    return Denhanced

def generate_score(D, Vmin, Vmax, nVel):
    """
    performs line search for each localisation in difference matrix which is 
    """
    N = D.shape[0]
    L = D.shape[1]

    velocities = np.linspace(Vmin, Vmax, nVel+1)
    times = np.arange(L) # t = 0, ..., L

    max_ind = int(N - 1 - Vmax * L) # last template image to begin sequence matching on
    refs = np.arange(max_ind) # i = 0, ..., max_ind <- truncated so line search not cut off
    optD = np.empty(max_ind) # D score for best velocity for each starting point (template image) 
    optD[:] = np.inf

    for vel in velocities:
        # indices in D for line search given a particular velocity
        row_indices = np.floor(refs[:, np.newaxis] + vel * times[np.newaxis, :]).astype(int).reshape(-1)
        col_indices = np.tile(times, max_ind)
        # evaluate D at indices and sum to get aggregate difference
        Dsum = np.sum(D[row_indices, col_indices].reshape(max_ind, L), axis=1)
        # for sequence matching scores better than prior scores (under different velocities), update
        ind_better = Dsum < optD 
        optD[ind_better] = Dsum[ind_better]
    return optD

def locate_best_match(scores, window):
    """
    Given a set of template scores, match
    """
    # indices of best match and window around it
    iOpt = np.argmin(scores)
    iWinL = np.maximum(iOpt - int(window / 2), 0)
    iWinU = np.minimum(iOpt + int(window / 2), len(scores))
    # check best match outside window
    outside_scores = np.concatenate((scores[:iWinL], scores[iWinU:]))
    optOutside = min(outside_scores)
    # for negative scores, u \in [0, 1] increases the score... adjust
    if optOutside > 0:
        mu = scores[iOpt] / optOutside
    else:
        mu = optOutside / scores[iOpt]
    return iOpt, mu

def localize_traverses(reference, query, desc, wContrast, numVel, vMin, vMax, matchWindow, enhance=False, verbose=False):
    ref_gt, ref_descriptors, _, query_gt, _, query_descriptors, _, _, _\
        = utils.import_traverses(reference, query, desc)
    nloc = len(query_descriptors)
    L = len(query_gt[0])

    scores = np.empty(nloc) # matching scores for each query seq
    proposals = []
    times = np.empty(nloc) # store compute times for each sequence

    for i in trange(len(query_descriptors), desc='SeqSLAM', leave=False):
        start = time.time() # time each query sequence
        # generate descriptor difference matrix
        D = np.empty((len(ref_descriptors), L))
        for t in range(L):
            D[:, t] = np.squeeze(np.linalg.norm(ref_descriptors[:, np.newaxis] - query_descriptors[i, t][np.newaxis, ...], axis=2))
        if enhance:
            D = contrast_enhancement(D, args.window_contrast) # enhance contrast
        # generate sequence matching scores for each template image
        ref_template_scores = generate_score(D, vMin, vMax, numVel) 
        ref_idx, score = locate_best_match(ref_template_scores, matchWindow) # quality score of best match
        # save proposals and scores
        scores[i] = score
        proposals.append(ref_gt[ref_idx])
        times[i] = time.time() - start
        if verbose:
            Trel = query_gt[i][0] / ref_gt[ref_idx]
            print("i", i, "dists", np.linalg.norm(Trel.t()), Trel.R().magnitude() * 180 / 3.1415, score)
    # process gt and proposal poses
    gt = [queries[0] for queries in query_gt]
    gt = geometry.combine(gt)
    proposals = geometry.combine(proposals)
    return proposals, scores, times, gt, L

def save_results(reference, query, proposals, scores, times, query_gt, L):
    model_name = 'SeqSLAM'
    # save model output
    save_path_query = os.path.join(utils.save_path, query)
    if not os.path.exists(save_path_query):
        os.makedirs(save_path_query)
    savefile = save_path_query + '/SeqSLAM.pickle'
    utils.save_obj(savefile, model=model_name, reference=args.reference_traverse, query=args.query_traverse, L=L, 
                                    query_gt=query_gt, proposals=proposals, scores=scores, times=times)

def main(args):
    if not args.All:
        proposals, scores, times, query_gt, L = localize_traverses(args.reference_traverse, args.query_traverse, args.descriptor, 
                            args.wContrast, args.numVel, args.vMin, args.vMax, args.matchWindow, enhance=args.enhance, verbose=args.verbose)
        save_results(args.reference_traverse, args.query_traverse, proposals, scores, times, query_gt, L)
    else:
        traverses = ['Rain', 'Night', 'Dusk']
        pbar = tqdm(traverses)
        for traverse in pbar:
            pbar.set_description(traverse)
            proposals, scores, times, query_gt, L = localize_traverses(args.reference_traverse, traverse, args.descriptor, 
                                args.wContrast, args.numVel, args.vMin, args.vMax, args.matchWindow, enhance=args.enhance, verbose=args.verbose)
            save_results(args.reference_traverse, traverse, proposals, scores, times, query_gt, L)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SeqSLAM on trials")
    parser.add_argument('-r', '--reference-traverse', type=str, default='2015-03-17-11-08-44',
                        help="reference traverse used to build map")
    parser.add_argument('-q', '--query-traverse', type=str, default='Night', choices=['Rain', 'Dusk', 'Night'],
                        help="query traverse to localize against reference map")
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
    parser.add_argument('-A', '--All', action='store_true', help='run all configurations required to generate results')
    parser.add_argument('--descriptor', type=str, default='NetVLAD', choices=['NetVLAD', 'DenseVLAD'], help='type of descriptor to use. options: NetVLAD')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
    args = parser.parse_args()

    main(args)