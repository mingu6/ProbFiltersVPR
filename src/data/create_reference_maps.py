import sys
import os
import argparse
import pickle
from tqdm import tqdm

import numpy as np

from src import geometry, utils, params
from src.thirdparty.robotcar_dataset_sdk import\
    interpolate_poses, transform


def build_reference_keyframes(gt, threshold, attitude_weight):
    indices = [0] # first image in set of keyframes
    gt_curr = gt[0]
    for i in range(1, len(gt) - 1):
        curr_diff = geometry.metric(gt_curr, gt[i], attitude_weight)
        if curr_diff > threshold:
            indices.append(i)
            gt_curr = gt[i]
    indices = np.asarray(indices)
    return indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract reference map"
                                     "keyframes as indices from"
                                     "processed raw traverses")
    parser.add_argument('-t', '--traverses', nargs='+', type=str,
                        default=['Overcast'],
                        help="Names of traverses to process,"
                        "e.g. Overcast, Night, Dusk etc. Input 'all'"
                        "instead to process all traverses."
                        "See src/params.py for full list.")
    parser.add_argument('-w', '--attitude-weight', type=float, default=15,
                        help="weight for attitude component of pose"
                        "distance equal to d where 1 / d being rotation"
                        "angle (rad) equivalent to 1m translation")
    parser.add_argument('-k', '--kf-threshold', type=float, default=0.5,
                        help="threshold on weighted pose distance to"
                        "generate new keyframe")
    args = parser.parse_args()

    if 'all' in args.traverses:
        names = params.traverses.keys()
    else:
        names = args.traverses

    pbar = tqdm(names)
    for name in pbar:
        pbar.set_description(name)
        # load full traverse data
        rtk_poses, _, descriptors, tstamps =\
            utils.load_traverse_data(name)
        # subsample traverse using increments based on RTK
        indices = \
            build_reference_keyframes(rtk_poses, args.kf_threshold,
                                      args.attitude_weight)
        rtk_ref = rtk_poses[indices]
        tstamps_ref = tstamps[indices]
        # save all to disk
        basepath = os.path.join(utils.reference_path,
                                params.traverses[name])
        rtkpath = os.path.join(basepath, 'rtk/stereo/left')
        if not os.path.exists(rtkpath):
            os.makedirs(rtkpath)
        descriptorpath = os.path.join(basepath,
                                      'descriptors/stereo/left')
        if not os.path.exists(descriptorpath):
            os.makedirs(descriptorpath)
        np.save(basepath + '/stereo_tstamps.npy', tstamps_ref)
        utils.save_obj(rtkpath + "/rtk.pickle", rtk=rtk_ref)
        for name, mat in descriptors.items():
            # descriptors in float32 for speed (float64 2x slower!)
            mat_ref = mat[indices].astype(np.float32)
            np.save(descriptorpath + '/{}.npy'.format(name), mat_ref)
