
import sys
import os
import argparse
import pickle
from multiprocessing import Pool

from tqdm import tqdm
import numpy as np
from sklearn.neighbors import BallTree
from scipy.spatial.transform import Rotation

from src import geometry, utils
from reference_traverse import build_reference_keyframes

"""
The following modules are imported from the robotcar-dataset-sdk
repo:

https://github.com/ori-mrg/robotcar-dataset-sdk

Please ensure this repo is cloned, with the "/python" folder added
to your PYTHONPATH before using these functions.
"""

import interpolate_poses, transform

################## filepaths ###################
file_dir = os.path.dirname(os.path.abspath(__file__))
raw_path = os.path.abspath(os.path.join(file_dir, "../../../", "data/raw/RobotCar/"))
processed_path = os.path.abspath(os.path.join(file_dir, "../../../", "data/processed/RobotCar/"))
################################################

def generate_random_localisations(start_indices, nloc, traverse_threshold, seq_len, start_threshold=0.5, seed=1):
    """
    Generates nloc random indices from raw query traverse to commence localisation.

    Args:
        indices (np array int N): Indices of raw query traverse images as possible starting locations for localisation
        nloc (int): Number of random localisations from starting indices.
        traverse_threshold (float): Threshold for generating a keyframe in traverse to be built.
                Used to ensure starting indices are not near end of full traverse and that building
                a sequence of length seq_len is possible.
        seq_len (int): Sequence length (in keyframes) to be built from startng indices.
        start_threshold (float): Threshold used to generate the starting indices set. 
    """
    rng = np.random.RandomState(seed=seed)
    ind_gap = int(traverse_threshold * seq_len / start_threshold) # last possible starting point must be far away enough to generate
                                                                  # a full sequence of len seq_len (given traverse_threshold) after initialisation.
    idx = rng.choice(np.arange(len(start_indices) - ind_gap), replace=False, size=nloc)
    return start_indices[np.sort(idx)]

def load_query_data(name):
    """
    Helper function to load processed traverse data from disk for a given
    traverse and descriptor combo for the purpose of constructing the query traverse.
    """
    with open(os.path.join(processed_path, name, "vo/vo.pickle"), 'rb') as f:
        vo = pickle.load(f)
    with open(os.path.join(processed_path, name, "rtk/rtk.pickle"), 'rb') as f:
        gt = pickle.load(f)
    return vo['cumulative'], gt['poses']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
            "Extract starting indices for query traverse and process traverses for random relocalisations")
    parser.add_argument('-t', '--traverses', nargs='+', type=str, required=True,
                        help="<Required> names of traverses to process, e.g. 2014-11-21-16-07-03 2015-03-17-11-08-44. \
                            Input 'all' instead to process all available raw traverses.")
    parser.add_argument('-w', '--attitude-weight', type=float, default=20, 
        help="weight for attitude components of pose distance equal to d where 1 / d being rotation angle (rad) equivalent to 1m translation")
    parser.add_argument('-ti', '--kf-threshold-indices', type=float, default=0.5, help='threshold on weighted pose distance to generate new starting index')
    parser.add_argument('-tt', '--kf-threshold-traverse', type=float, default=3, help='threshold on weighted pose distance for traverse from starting index')
    parser.add_argument('-L', '--seq-len', type=int, default=30, help='number of keyframes in sequence for each trial relocalisation')
    parser.add_argument('-n', '--nloc', type=int, default=500, help='number of relocalisations to perform from generated indices')
    parser.add_argument('-d', '--descriptor', type=str, default='NetVLAD', help='type of descriptor to use. options: NetVLAD')
    parser.add_argument('-j', '--jobs', type=int, default=4, help='number of parallel processes spawned')
    args = parser.parse_args()

    descriptors = ['NetVLAD', 'DenseVLAD']
    if args.descriptor not in descriptors:
        parser.error('Invalid descriptor type for --descriptor. Valid options are: {}'.format(", ".join(descriptors)))

    if 'all' in args.traverses:
        names = [f for f in os.listdir(raw_path) if f.startswith("201")]
    else:
        names = args.traverses

    for name in tqdm(names, desc='traverses'):
        # use VO to generate possible starting indices for localisation. VO is used to ensure
        vo, gt = load_query_data(name)
        start_indices = build_reference_keyframes(gt, args.kf_threshold_indices, args.attitude_weight)
        random_indices = generate_random_localisations(start_indices, args.nloc, args.kf_threshold_traverse, args.seq_len, args.kf_threshold_indices)
        # from starting indices, generate finite length traverses 
        voQueries = []
        gtMotions = []
        indices_full = []
        for start_id in random_indices:
            w = args.attitude_weight
            T = args.kf_threshold_traverse
            L = args.seq_len

            indices = [start_id]
            voQuery = []
            gtMotion = []

            recent_id = start_id # id of most recent keyframe
            curr_id = start_id + 1 # current active frame
            while len(indices) < L:
                dist = geometry.metric(gt[recent_id], gt[curr_id], w)
                if dist > T:
                    indices.append(curr_id)
                    voQuery.append(vo[recent_id] / vo[curr_id])
                    gtMotion.append(gt[recent_id] / gt[curr_id])
                    gt_curr = gt[curr_id] # update pose of new keyframe as current
                    recent_id = curr_id
                curr_id += 1
            voQueries.append(geometry.combine(voQuery))
            gtMotions.append(geometry.combine(gtMotion))
            discr = geometry.metric(vo[start_id] / vo[recent_id], gt[start_id] / gt[recent_id], w)
            indices_full.append(np.asarray(indices))
        

        savepath = os.path.join(processed_path, name, "query")
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        with open(savepath + '/indices.pickle', 'wb') as f:
            pickle.dump(start_indices, f)

        utils.save_obj(savepath + '/traverses.pickle', vo=voQueries, gt_motion=gtMotions,
                        indices=np.asarray(indices_full), w=args.attitude_weight)