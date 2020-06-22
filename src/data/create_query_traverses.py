
import sys
import os
import argparse
import pickle
from multiprocessing import Pool

from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation

from src import geometry, utils, params
from src.thirdparty.robotcar_dataset_sdk import interpolate_poses, transform
from create_reference_maps import build_reference_keyframes

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
            "Extract starting indices for query traverse and process traverses for random relocalisations")
    parser.add_argument('-t', '--traverses', nargs='+', type=str, default=['Rain', 'Dusk', 'Night'],
                        help="Names of traverses to process, e.g. Overcast, Night, Dusk etc. \
                            Input 'all' instead to process all traverses. See src/params.py for full list.")
    parser.add_argument('-w', '--attitude-weight', type=float, default=20, 
        help="weight for attitude components of pose distance equal to d where 1 / d being rotation angle (rad) equivalent to 1m translation")
    parser.add_argument('-ti', '--kf-threshold-indices', type=float, default=0.5, help='threshold on weighted pose distance to generate new starting index')
    parser.add_argument('-tt', '--kf-threshold-traverse', type=float, default=3, help='threshold on weighted pose distance for traverse from starting index')
    parser.add_argument('-L', '--seq-len', type=int, default=30, help='number of keyframes in sequence for each trial relocalisation')
    parser.add_argument('-n', '--nloc', type=int, default=500, help='number of relocalisations to perform from generated indices')
    args = parser.parse_args()

    w = args.attitude_weight
    T = args.kf_threshold_traverse
    L = args.seq_len

    if 'all' in args.traverses:
        names = params.traverses.keys()
    else:
        names = args.traverses

    pbar = tqdm(names)
    for name in pbar:
        pbar.set_description(name)
        # load full traverse data
        rtk_poses, vo_cumulative, descriptors, tstamps = utils.load_traverse_data(name) 
        start_indices = build_reference_keyframes(rtk_poses, args.kf_threshold_indices, args.attitude_weight)
        random_indices = generate_random_localisations(start_indices, args.nloc, args.kf_threshold_traverse, 
                        args.seq_len, args.kf_threshold_indices)
        # from starting indices, generate finite length traverses 
        voQueries = []
        rtkMotions = []
        rtkPoses = []
        descriptors_full = dict((name, []) for name in descriptors.keys())

        tstamps_full = []
        for start_id in random_indices:
            # store output for a single sequence
            indices = [start_id]
            voQuery = []
            rtkMotion = []

            recent_id = start_id # id of most recent keyframe
            curr_id = start_id + 1 # current active frame
            while len(indices) < L:
                dist = geometry.metric(rtk_poses[recent_id], rtk_poses[curr_id], w)
                if dist > T:
                    indices.append(curr_id)
                    voQuery.append(vo_cumulative[recent_id] / vo_cumulative[curr_id])
                    rtkMotion.append(rtk_poses[recent_id] / rtk_poses[curr_id])
                    rtk_curr = rtk_poses[curr_id] # update pose of new keyframe as current
                    recent_id = curr_id
                curr_id += 1
            indices = np.asarray(indices)
            voQueries.append(geometry.combine(voQuery))
            rtkMotions.append(geometry.combine(rtkMotion))
            rtkPoses.append(rtk_poses[indices])
            # for each descriptor type, add sequence to list
            for desc, mat in descriptors.items():
                descriptors_full[desc].append(mat[indices].astype(np.float32)) # descriptors in float32 for speed (float64 2x slower!)
            tstamps_full.append(tstamps[indices])

        # save all to disk
        savepath = os.path.join(utils.query_path, params.traverses[name])
        rtkpath = os.path.join(utils.query_path, params.traverses[name], 'rtk/stereo/left')
        if not os.path.exists(rtkpath): 
            os.makedirs(rtkpath)
        descriptor_path = os.path.join(utils.query_path, params.traverses[name], 'descriptors/stereo/left') 
        if not os.path.exists(descriptor_path): 
            os.makedirs(descriptor_path)
        np.save(savepath + '/stereo_tstamps.npy', tstamps_full)
        utils.save_obj(rtkpath + "/rtk.pickle", rtk=rtkPoses)
        utils.save_obj(savepath + "/vo.pickle", odom=voQueries)
        utils.save_obj(savepath + "/rtk_motion.pickle", odom=rtkMotions)
        for name, mat in descriptors_full.items():
            np.save(descriptor_path + '/{}.npy'.format(name), np.asarray(mat))