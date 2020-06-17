import sys
import os
import argparse
import pickle
from tqdm import tqdm

import numpy as np

from src import geometry, utils

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

def build_reference_keyframes(gt, threshold, attitude_weight):
    """
    Generates traverse using VO by saving indices corresponding to keyframes. Also
    saved relative pose between keyframes from VO.

    NOTE: First element of motion array (vo) is relative motion between first two keyframes.

    Args:
        W (np array float 6): Weight for each component of pseudo-se3 representation
                    of relative pose from VO when calculating distance in manifold.
        threshold (float > 0): Threshold on weighted distance of accumulated VO from
                    current keyframe to be reached before creating a new keyframe.
        vo (array (N-1)x4x4): Processed VO data, where N is the number of images in traverse.
        descriptors (array NxD): Visual descriptor array for full traverse.
        gt_poses_pse3 (array Nx6): Pseudo-se(3) form of ground truth poses of images in traverse.
    """

    indices = [0] # first image in set of keyframes
    gt_curr = gt[0]
    for i in range(1, len(gt) - 1):
        curr_diff = geometry.metric(gt_curr, gt[i], attitude_weight)
        if curr_diff > threshold:
            indices.append(i)
            gt_curr = gt[i]
    indices = np.asarray(indices)
    return indices

def load_reference_data(name):
    """
    Helper function to load processed traverse data from disk for a given
    traverse and descriptor combo for the purpose of constructing the reference traverse.
    """
    with open(os.path.join(processed_path, name, "rtk/rtk.pickle"), 'rb') as f:
        gt = pickle.load(f)
    return gt['poses']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
            "Extract reference map keyframes as indices from processed raw traverses")
    parser.add_argument('-t', '--traverses', nargs='+', type=str, required=True,
                        help="<Required> names of traverses to process, e.g. 2014-11-21-16-07-03 2015-03-17-11-08-44. \
                            Input 'all' instead to process all available raw traverses.")
    parser.add_argument('-w', '--attitude-weight', type=float, default=20, 
        help="weight for attitude components of pose distance equal to d where 1 / d being rotation angle (rad) equivalent to 1m translation")
    parser.add_argument('-k', '--kf-threshold', type=float, default=0.5, help='threshold on weighted pose distance to generate new keyframe')
    args = parser.parse_args()

    if args.traverses[0] == 'all':
        names = [f for f in os.listdir(raw_path) if f.startswith("201")]
    else:
        names = args.traverses

    for name in tqdm(names):
        gt_full = load_reference_data(name)
        indices = build_reference_keyframes(gt_full, args.kf_threshold, args.attitude_weight)
        savepath = os.path.join(processed_path, name, "reference")
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        utils.save_obj(savepath + "/indices.pickle", indices=indices, w=args.attitude_weight)
