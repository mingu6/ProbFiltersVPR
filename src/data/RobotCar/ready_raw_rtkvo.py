import sys
import os
import argparse

import pandas as pd
import numpy as np
import pickle

from scipy.spatial.transform import Rotation
from tqdm import tqdm

from src import geometry
from query_traverse import save_obj
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

def process_raw_traverse(name, min_idx):
    """
    Given a traverse name, process raw RTK and VO data by interpolating it at all
    image timestamp values to approximate both ground truth pose and relative pose 
    between images using VO data. Let N be the number of images.
    
    Args:
        name (str): name of traverse, e.g. 2014-11-21-16-07-03

    Returns:
        tstamps (numpy array int N): sorted image timestamps
        gt_poses_SE3 (numpy array float Nx4x4): interpolated ground truth RTK pose 
                        represented as an SE(3) 4x4 matrix element for each image
        gt_poses_pse3 (numpy array float Nx6) : as above, but represented as a pseudo-se3
                        element consisting of a translation vector and rotation vector 
                        representing the rotation axis. The norm of the rotation vector is equal
                        to the angle of rotation around the rotation axis.
        vo_relative (numpy array float Nx4x4): set of 4x4 matrices each representing a rigid
                        body transformation between images. The second element is the relative
                        transformation between the first and second image.
    """
    raw_dir = os.path.join(raw_path, name)
    vo_path = raw_dir + '/vo/vo.csv'
    rtk_path = raw_dir + '/rtk/rtk.csv'

    def tstamp(fname):
        """
        helper function to sort images by timestamp value
        """
        return int(fname[:-4])

    # sort images by timestamp order
    img_filenames = [f for f in os.listdir(raw_dir + '/stereo/left/') if f.endswith('.png')]
    img_filenames.sort(key=tstamp)
    img_tstamps = [int(fname[:-4]) for fname in img_filenames]
    img_tstamps_np = np.asarray(img_tstamps[min_idx:])

    # Load VO data and estimate incremental SE(3) poses between images from VO.
    # Note that VO after interpolation is cumulative from origin frame, and so
    # relative pose needs to be computed.

    vo_cumulative = interpolate_poses.interpolate_vo_poses(vo_path, 
            img_tstamps, img_tstamps[min_idx])
    vo_cumulative = np.asarray(vo_cumulative)
    vo = geometry.SE3Poses(vo_cumulative[min_idx:, :3, 3], Rotation.from_dcm(vo_cumulative[min_idx:, :3, :3]))

    # extract absolute gt pose for first image for pose
    # w.r.t. world reference frame (northing, easting)
    rtk = pd.read_csv(rtk_path)
    rtk_tstamps = rtk[['timestamp']].to_numpy()
    idx = np.abs(rtk_tstamps - img_tstamps[min_idx]).argmin()
    # read in raw position data)
    xyzrpy = rtk[['northing', 'easting', 'down', 'roll', 'pitch', 'yaw']].iloc[idx].to_numpy()

    T0 = np.array(transform.build_se3_transform(xyzrpy))

    # TO DO: Sensor intrinsics!!!
    # load estimated relative ground truth poses for each image
    rel_gt_poses = interpolate_poses.interpolate_ins_poses(rtk_path, 
            img_tstamps, img_tstamps[min_idx], use_rtk=True)
    # rel_gt_poses = [rel @ Tins for rel in rel_gt_poses]
    rel_gt_poses = np.asarray(rel_gt_poses)

    gt_mat = T0[np.newaxis, :] @ rel_gt_poses[min_idx:]
    # adjust poses so VO and RTK are aligned in coordinate frame 
    R = np.array([[ 0, 1, 0],
                  [-1, 0, 0],
                  [ 0, 0, 1]])
    gt_mat[:, :3, :3] = R[np.newaxis, ...] @ gt_mat[:, :3, :3]
    gt = geometry.SE3Poses(gt_mat[min_idx:, :3, 3], Rotation.from_dcm(gt_mat[min_idx:, :3, :3]))

    return img_tstamps_np, gt, vo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
            "Extract ground truth poses and motion commands from RobotCar dataset")
    parser.add_argument('-t', '--traverses', nargs='+', type=str, required=True,
                        help="<Required> names of traverses to process, e.g. 2014-11-21-16-07-03 2015-03-17-11-08-44. \
                            Input 'all' instead to process all available raw traverses.")
    args = parser.parse_args()
    if args.traverses[0] == 'all':
        names = [f for f in os.listdir(raw_path) if f.startswith("201")]
    else:
        names = args.traverses

    min_idx = 0 # remove first few frames in sequence for GPS to initialize

    for name in tqdm(names):
        # extract ground truth poses and VO from raw
        tstamps, gt, vo = process_raw_traverse(name, min_idx)

        # save to data/processed/RobotCar/name/ directory, create if they do not exist
        base_dir = os.path.join(processed_path, name)
        rtk_path = os.path.join(base_dir, "rtk")
        vo_path = os.path.join(base_dir, "vo")

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        if not os.path.exists(rtk_path):
            os.makedirs(rtk_path)

        if not os.path.exists(vo_path):
            os.makedirs(vo_path)

        np.save(base_dir + '/tstamps.npy', tstamps)
        save_obj(rtk_path + '/rtk.pickle', poses=gt)
        save_obj(vo_path + '/vo.pickle', cumulative=vo)