import os
import pickle
import numpy as np

from . import params

################## filepaths ###################
curr_path = os.path.dirname(os.path.realpath(__file__))
raw_path = "/home/ming/data/raw/RobotCar/"
processed_path = "/home/ming/data/processed/RobotCar/"
reference_path = os.path.abspath(os.path.join(curr_path, "..", "data/reference"))
query_path = os.path.abspath(os.path.join(curr_path, "..", "data/query"))
results_path = os.path.abspath(os.path.join(curr_path, "..", "results"))
################################################

def save_obj(savepath, **components):
    with open(savepath, 'wb') as f:
        pickle.dump(components, f)
    return save_obj
        
def load_traverse_data(name):
    """
    Helper function to load processed traverse data from disk for a given
    traverse and descriptor combo for the purpose of constructing the reference traverse.
    """
    # import timestamps
    tstamps_dir = os.path.join(processed_path, params.traverses[name], "stereo_tstamps.npy")
    tstamps = np.load(tstamps_dir)

    # import RTK GPS poses for images
    rtk_dir = os.path.join(processed_path, params.traverses[name], "rtk/stereo/left/rtk.pickle")
    with open(rtk_dir, 'rb') as f:
        rtk = pickle.load(f)
    rtk_poses = rtk['rtk'] 

    # import VO
    vo_dir = os.path.join(processed_path, params.traverses[name], "vo/vo.pickle")
    with open(vo_dir, 'rb') as f:
        vo = pickle.load(f)
    vo_cumulative = vo['cumulative'] 

    # import all available image descriptors
    descriptors_dir = os.path.join(processed_path, params.traverses[name], "stereo/left/")
    descriptors = {}
    for fname in os.listdir(descriptors_dir):
        if fname.endswith('.npy'):
            descriptorName = fname[:-4]
            descriptorMat = np.load(os.path.join(descriptors_dir, fname))
            descriptors[descriptorName] = descriptorMat
    return rtk_poses, vo_cumulative, descriptors, tstamps

def import_traverses(ref_name, query_name, descriptor, gt_motion=False):
    # load full processed traverse
    ref_path = os.path.join(processed_path, ref_name)
    query_path = os.path.join(processed_path, params.queries[query_name])

    # import full processed traverses, subsample for map and query
    with open(ref_path + '/rtk/rtk.pickle', 'rb') as f:
        ref_gt_full =  pickle.load(f)['poses']
    ref_descriptors_full = np.load(ref_path + '/stereo/left/{}.npz'.format(descriptor))['arr_0']
    query_descriptors = np.load(query_path + '/stereo/left/{}.npz'.format(descriptor))['arr_0']

    # load reference and query indices
    with open(query_path + '/query/traverses.pickle', 'rb') as f:
        query = pickle.load(f)
    with open(query_path + '/rtk/rtk.pickle', 'rb') as f:
        query_gt_full = pickle.load(f)['poses']
    query_traverse_ind = query['indices']
    query_gt = [query_gt_full[ind] for ind in query_traverse_ind]
    # load query odometry
    if gt_motion:
        query_vo = query["gt_motion"]
    else:
        query_vo = query["vo"]
    w = query['w']
    # load query ground truth poses
    query_tstamps = np.load(query_path + '/tstamps.npy')
    traverses_tstamps = query_tstamps[query_traverse_ind]
    traverses_descriptors = query_descriptors[query_traverse_ind]
    # import reference traverses
    with open(ref_path + '/reference/indices.pickle', 'rb') as f:
        reference = pickle.load(f)
    ref_kf_ind = reference["indices"]
    ref_gt = ref_gt_full[ref_kf_ind]
    ref_descriptors = ref_descriptors_full[ref_kf_ind]
    ref_tstamps_full = np.load(ref_path + '/tstamps.npy')
    ref_tstamps = ref_tstamps_full[ref_kf_ind]
    
    return ref_gt, ref_descriptors, query_traverse_ind, query_gt, query_vo, traverses_descriptors, ref_tstamps, traverses_tstamps, w
