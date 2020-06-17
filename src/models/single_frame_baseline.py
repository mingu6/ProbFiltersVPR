
import os
import argparse
import pickle
import time

import numpy as np
from tqdm import trange, tqdm

from src import geometry, utils

################## filepaths ###################
file_dir = os.path.dirname(os.path.abspath(__file__))
raw_path = os.path.abspath(os.path.join(file_dir, "../../", "data/raw/RobotCar/"))
processed_path = os.path.abspath(os.path.join(file_dir, "../../", "data/processed/RobotCar/"))
save_path = os.path.abspath(os.path.join(file_dir, "../../", "models"))
################################################

def localize_traverses(reference, query, desc):
    ref_gt, ref_descriptors, _, query_gt, _, query_descriptors, _, _, _\
        = utils.import_traverses(reference, query, desc)
    # experiment parameters
    nloc = len(query_gt)
    # store model output
    scores = np.empty(nloc)
    proposals = []
    times = np.empty(nloc) # store compute times for each sequence
    for i in trange(nloc, desc='Single', leave=False):
        start = time.time() # time iteration
        dists = np.linalg.norm(query_descriptors[i, 0].reshape(1, -1) - ref_descriptors, axis=1)
        idx = np.argmin(dists)
        # save localization result and data
        times[i] = time.time() - start
        scores[i] = dists[idx]
        proposals.append(ref_gt[idx])

    # process gt and proposal poses
    gt = [queries[0] for queries in query_gt]
    gt = geometry.combine(gt)
    proposals = geometry.combine(proposals)
    return proposals, scores, times, gt

def save_results(reference, query, proposals, scores, times, query_gt):
    # save model output
    save_path_query = os.path.join(utils.save_path, query)
    if not os.path.exists(save_path_query):
        os.makedirs(save_path_query)
    model_name = 'Single'
    savefile = save_path_query + '/Single.pickle'
    utils.save_obj(savefile, model=model_name, reference=args.reference_traverse, query=args.query_traverse, query_gt=query_gt, proposals=proposals, scores=scores, times=times)

def main(args):
    if not args.All:
        proposals, scores, times, query_gt = localize_traverses(args.reference_traverse, args.query_traverse, args.descriptor)
        save_results(args.reference_traverse, args.query_traverse, proposals, scores, times, query_gt)
    else:
        traverses = ['Rain', 'Night', 'Dusk']
        pbar = tqdm(traverses)
        for traverse in pbar:
            pbar.set_description(traverse)
            proposals, scores, times, query_gt = localize_traverses(args.reference_traverse, traverse, args.descriptor)
            save_results(args.reference_traverse, traverse, proposals, scores, times, query_gt)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single image matching baseline")
    parser.add_argument('-r', '--reference-traverse', type=str, default='2015-03-17-11-08-44',
                        help="reference traverse used to build map")
    parser.add_argument('-q', '--query-traverse', type=str, default='Night', choices=['Rain', 'Dusk', 'Night'],
                        help="query traverse to localize against reference map")
    parser.add_argument('--descriptor', type=str, default='NetVLAD', choices=['NetVLAD', 'DenseVLAD'], help='type of descriptor to use. options: NetVLAD')
    parser.add_argument('-A', '--All', action='store_true', help='run all configurations required to generate results')
    args = parser.parse_args()

    descriptors = ['NetVLAD', 'DenseVLAD']
    if args.descriptor not in descriptors:
        parser.error('Invalid descriptor type for --descriptor. Valid options are: {}'.format(", ".join(descriptors)))

    main(args)