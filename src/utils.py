import os
import pickle
import time

import numpy as np
from tqdm import tqdm, trange

from src import params, geometry
from src.settings import RAW_PATH, READY_PATH, PROCESSED_PATH

################ DO NOT MODIFY BELOW ###############
curr_path = os.path.dirname(os.path.realpath(__file__))
reference_path = os.path.abspath(os.path.join(curr_path, "..", "data/reference"))
query_path = os.path.abspath(os.path.join(curr_path, "..", "data/query"))
results_path = os.path.abspath(os.path.join(curr_path, "..", "results"))
figures_path = os.path.abspath(os.path.join(curr_path, "..", "figures"))
models_path = os.path.abspath(
    os.path.join(curr_path, "thirdparty/robotcar_dataset_sdk/models")
)
####################################################


def save_obj(savepath, **components):
    with open(savepath, "wb") as f:
        pickle.dump(components, f)
    return save_obj


def load_traverse_data(name):
    """
    Helper function to load processed traverse data from disk for a given
    traverse and descriptor combo for the purpose of constructing the
    reference traverse.
    """
    # import timestamps
    tstamps_dir = os.path.join(
        PROCESSED_PATH, params.traverses[name], "stereo_tstamps.npy"
    )
    tstamps = np.load(tstamps_dir)

    # import RTK GPS poses for images
    rtk_dir = os.path.join(
        PROCESSED_PATH, params.traverses[name], "rtk/stereo/left/rtk.pickle"
    )
    with open(rtk_dir, "rb") as f:
        rtk_poses = pickle.load(f)["rtk"]

    # import VO
    vo_dir = os.path.join(PROCESSED_PATH, params.traverses[name], "vo/vo.pickle")
    with open(vo_dir, "rb") as f:
        vo_cumulative = pickle.load(f)["cumulative"]

    # import all available image descriptors
    descriptors_dir = os.path.join(
        PROCESSED_PATH, params.traverses[name], "stereo/left/"
    )
    descriptors = {}
    for fname in os.listdir(descriptors_dir):
        if fname.endswith(".npy"):
            descriptorName = fname[:-4]
            descriptorMat = np.load(os.path.join(descriptors_dir, fname))
            descriptors[descriptorName] = descriptorMat
    return rtk_poses, vo_cumulative, descriptors, tstamps


def import_query_traverse(name):
    # import timestamps
    base_dir = os.path.join(query_path, params.traverses[name])
    tstamps = np.load(base_dir + "/stereo_tstamps.npy")
    # import RTK GPS poses for images
    rtk_dir = os.path.join(base_dir, "rtk/stereo/left/rtk.pickle")
    with open(rtk_dir, "rb") as f:
        rtk_poses = pickle.load(f)["rtk"]
    # import VO and RTK motion
    with open(base_dir + "/vo.pickle", "rb") as f:
        vo = pickle.load(f)["odom"]
    with open(base_dir + "/rtk_motion.pickle", "rb") as f:
        rtk_motion = pickle.load(f)["odom"]
    # import all available image descriptors
    descriptors_dir = os.path.join(
        query_path, params.traverses[name], "descriptors/stereo/left/"
    )
    descriptors = {}
    for fname in os.listdir(descriptors_dir):
        if fname.endswith(".npy"):
            descriptorName = fname[:-4]
            descriptorMat = np.load(os.path.join(descriptors_dir, fname))
            descriptors[descriptorName] = descriptorMat
    return rtk_poses, vo, rtk_motion, descriptors, tstamps


def import_reference_map(name):
    # import timestamps
    tstamps_dir = os.path.join(
        reference_path, params.traverses[name], "stereo_tstamps.npy"
    )
    tstamps = np.load(tstamps_dir)
    # import RTK GPS poses for images
    rtk_dir = os.path.join(
        reference_path, params.traverses[name], "rtk/stereo/left/rtk.pickle"
    )
    with open(rtk_dir, "rb") as f:
        rtk = pickle.load(f)
    rtk_poses = rtk["rtk"]
    # import all available image descriptors
    descriptors_dir = os.path.join(
        reference_path, params.traverses[name], "descriptors/stereo/left/"
    )
    descriptors = {}
    for fname in os.listdir(descriptors_dir):
        if fname.endswith(".npy"):
            descriptorName = fname[:-4]
            descriptorMat = np.load(os.path.join(descriptors_dir, fname))
            descriptors[descriptorName] = descriptorMat
    return rtk_poses, descriptors, tstamps


def localize_traverses_filter(model, query_descriptors, gt=None, vo=None, desc=None):
    nloc = len(query_descriptors)  # number of localizations
    L = len(query_descriptors[0])  # max sequence length
    # localize trials!
    proposals, scores, times = [], [], []
    for i in trange(nloc, desc=desc, leave=False):
        # save outputs of model
        proposals_seq = []
        score_seq = np.empty(L)
        times_seq = np.empty(L)

        start = time.time()  # begin timing localization
        model.initialize_model(query_descriptors[i, 0, :])

        for t in range(L):
            if t > 0:
                if vo:
                    model.update(vo[i][t - 1], query_descriptors[i, t, :])
                else:
                    model.update(query_descriptors[i, t, :])
            proposal, score = model.localize()
            proposals_seq.append(proposal)
            score_seq[t] = score
            times_seq[t] = time.time() - start
            # check results
        proposals_seq = geometry.combine(proposals_seq)
        # save traverse statistics
        proposals.append(proposals_seq)
        scores.append(score_seq)
        times.append(times_seq)
    return proposals, scores, times


def localize_traverses_matching(
    model, query_poses, query_descriptors, desc=None, idx=0
):
    nloc = len(query_descriptors)  # number of localizations
    # localize trials!
    proposals, scores, times, query_gts = [], [], [], []
    for i in trange(nloc, desc=desc, leave=False):
        start = time.time()  # begin timing localization
        proposal, score = model.localize(query_descriptors[i])
        times.append(time.time() - start)
        proposals.append(proposal)
        scores.append(score)
        query_gts.append(query_poses[i][idx])
    return proposals, scores, times, query_gts


def remove_far_queries(ref_tree, ref_poses, query_poses):
    # import matplotlib.pyplot as plt
    divergence = []
    for i in range(len(query_poses)):
        query_seq = query_poses[i]
        dist, ind = ref_tree.nearest(query_seq.t(), query_seq.R().as_quat(), 1, 1)
        divergence.append(np.sum(dist))

    divergence = np.asarray(divergence)
    ind_not_outlier = np.squeeze(np.argwhere(divergence < params.outlier_threshold))
    return ind_not_outlier
