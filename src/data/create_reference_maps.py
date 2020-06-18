import sys
import os
import argparse
import pickle
from tqdm import tqdm

import numpy as np

from src import geometry, utils, params
from src.thirdparty.robotcar_dataset_sdk import interpolate_poses, transform

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
            "Extract reference map keyframes as indices from processed raw traverses")
    parser.add_argument('-t', '--traverses', nargs='+', type=str, default=['all'],
                        help="Names of traverses to process, e.g. Overcast, Night, Dusk etc. \
                            Input 'all' instead to process all traverses. See src/params.py for full list.")
    parser.add_argument('-w', '--attitude-weight', type=float, default=20, 
        help="weight for attitude components of pose distance equal to d where 1 / d being rotation angle (rad) equivalent to 1m translation")
    parser.add_argument('-k', '--kf-threshold', type=float, default=0.5, help='threshold on weighted pose distance to generate new keyframe')
    args = parser.parse_args()

    if 'all' in args.traverses:
        names = params.traverses.keys()
    else:
        names = args.traverses

    pbar = tqdm(names)
    for name in pbar:
        pbar.set_description(name)
        # load full traverse data
        rtk_poses, _, descriptors, tstamps = utils.load_traverse_data(name) 
        # subsample traverse using increments based on RTK
        indices = build_reference_keyframes(rtk_poses, args.kf_threshold, args.attitude_weight)
        rtk_ref = rtk_poses[indices]
        tstamps_ref = tstamps[indices]
        # save all to disk
        rtkpath = os.path.join(utils.reference_path, params.traverses[name], 'rtk/stereo/left')
        if not os.path.exists(rtkpath): 
            os.makedirs(rtkpath)
        descriptor_path = os.path.join(utils.reference_path, params.traverses[name], 'descriptors/stereo/left') 
        if not os.path.exists(descriptor_path): 
            os.makedirs(descriptor_path)
        np.save(rtkpath + '/stereo_tstamps.npy', tstamps_ref)
        utils.save_obj(rtkpath + "/rtk.pickle", rtk=rtk_ref)
        for name, mat in descriptors.items():
            mat_ref = mat[indices]
            np.save(descriptor_path + '/{}.npy'.format(name), mat_ref)