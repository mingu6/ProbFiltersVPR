import sys
import os
import argparse

import pandas as pd
import numpy as np
import pickle

from scipy.spatial.transform import Rotation
from tqdm import tqdm

from src import geometry, utils, params
from src.thirdparty.robotcar_dataset_sdk import\
    interpolate_poses, transform
from src.settings import RAW_PATH


def process_raw_traverse(name):
    raw_dir = os.path.join(RAW_PATH, params.traverses[name])
    vo_path = raw_dir + '/vo/vo.csv'
    rtk_path = raw_dir + '/rtk.csv'

    def tstamp(fname):
        """
        helper function to sort images by timestamp value
        """
        return int(fname[:-4])

    # sort images by timestamp order
    img_filenames = [f for f in os.listdir(raw_dir + '/stereo/left/')
                     if f.endswith('.png')]
    img_filenames.sort(key=tstamp)
    img_tstamps = [int(fname[:-4]) for fname in img_filenames]

    # Load VO data and estimate incremental SE(3) poses
    # between images from VO.
    # Note that VO after interpolation is cumulative from
    # origin frame, and so
    # relative pose needs to be computed.

    vo_cumulative =\
        interpolate_poses.interpolate_vo_poses(vo_path, img_tstamps,
                                               img_tstamps[0])
    vo_cumulative = np.asarray(vo_cumulative)
    vo = geometry.SE3Poses(vo_cumulative[:, :3, 3],
                           Rotation.from_dcm(vo_cumulative[:, :3, :3]))

    # extract absolute gt pose for first image for pose
    # w.r.t. world reference frame (northing, easting)
    rtk = pd.read_csv(rtk_path)
    rtk_tstamps = rtk[['timestamp']].to_numpy()
    idx = np.abs(rtk_tstamps - img_tstamps[0]).argmin()
    # read in raw position data)
    xyzrpy = rtk[['northing', 'easting', 'down',
                  'roll', 'pitch', 'yaw']].iloc[idx].to_numpy()

    T0 = np.array(transform.build_se3_transform(xyzrpy))

    # TO DO: Sensor intrinsics!!!
    # load estimated relative ground truth poses for each image
    rel_rtk_poses = \
        interpolate_poses.interpolate_ins_poses(
            rtk_path, img_tstamps, img_tstamps[0], use_rtk=True)
    rel_rtk_poses = np.asarray(rel_rtk_poses)

    rtk_mat = T0[np.newaxis, :] @ rel_rtk_poses
    # adjust poses so VO and RTK are aligned in coordinate frame 
    R = np.array([[ 0, 1, 0],
                  [-1, 0, 0],
                  [ 0, 0, 1]])
    rtk_mat[:, :3, :3] = R[np.newaxis, ...] @ rtk_mat[:, :3, :3]
    rtk_interp = geometry.SE3Poses(rtk_mat[:, :3, 3],
                                   Rotation.from_dcm(rtk_mat[:, :3, :3]))

    return rtk_interp, vo, np.asarray(img_tstamps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ground truth"
                                     "poses and motion commands"
                                     "from RobotCar dataset")
    parser.add_argument('-t', '--traverses', nargs='+', type=str,
                        default=['all'],
                        help="Names of traverses to process, e.g."
                        "Overcast, Night, Dusk etc. Input 'all' instead"
                        "to process all traverses. See src/params.py for"
                        "full list.")
    args = parser.parse_args()

    if 'all' in args.traverses:
        names = params.traverses.keys()
    else:
        names = args.traverses

    pbar = tqdm(names)
    for name in pbar:
        pbar.set_description(name)
        # extract ground truth poses and VO from raw
        rtk, vo, tstamps = process_raw_traverse(name)

        # TO DO: Handle other cameras
        base_dir = os.path.join(utils.processed_path,
                                params.traverses[name])
        rtk_path = os.path.join(base_dir, "rtk/stereo/left")
        vo_path = os.path.join(base_dir, "vo")

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        if not os.path.exists(rtk_path):
            os.makedirs(rtk_path)

        if not os.path.exists(vo_path):
            os.makedirs(vo_path)

        np.save(base_dir + '/stereo_tstamps.npy', tstamps)
        utils.save_obj(rtk_path + '/rtk.pickle', rtk=rtk)
        utils.save_obj(vo_path + '/vo.pickle', cumulative=vo)
