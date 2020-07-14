import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from src import utils, geometry, params
from src.thirdparty.robotcar_dataset_sdk import image, camera_model


def main(args):
    # error tolerances
    t_tol = np.array([3., 5.])
    R_tol = np.array([15., 30.])
    # which reference images to display
    # indices = np.asarray([30, 4500, 6953, 12100, 17000, 25170])
    indices = np.asarray([30, 6953, 25170])
    # import reference traverse data 
    ref_poses, _, _, ref_tstamps = utils.load_traverse_data(args.reference)
    # initialize plots
    nRows = len(args.queries)
    nCols = len(t_tol) + 1
    fig, axs = plt.subplots(nRows, nCols, figsize=(nCols * 6, nRows * 5))
    for i in range(nRows):
        # import correct query traverse for row
        traverse = args.queries[i]
        query_poses, _, _, query_tstamps =\
            utils.load_traverse_data(traverse)
        # retrieve reference pose
        indR = indices[i]
        ref_pose = ref_poses[indR]
        for j in range(nCols):
            if i == 0 and j == 0:
                axs[i, j].set_title("Reference", fontsize=32)
            elif i == 0:
                axs[i, j].set_title(
                    "{:.0f}m {:.0f} deg".format(t_tol[j-1], R_tol[j-1]),
                    fontsize=32
                )
            if j == 0:
                imgFolder = os.path.join(
                    utils.raw_path, params.traverses[args.reference],
                    'stereo/left')
                imgPath = os.path.join(imgFolder,
                                       str(ref_tstamps[indR]) + '.png')
            else:
                # compute all query distances to reference pose
                Trel = query_poses / ref_pose
                t_dist = np.linalg.norm(Trel.t(), axis=1)
                R_dist = Trel.R().magnitude() * 180 / np.pi
                SE3dist = geometry.metric(query_poses, ref_pose, 20)
                # locate all queries inside of tolerance
                tOK = t_dist < t_tol[j-1]
                ROK = R_dist < R_tol[j-1]
                bothOK = np.argwhere(np.logical_and(tOK, ROK))
                indQ = np.squeeze(bothOK[np.argmax(SE3dist[bothOK])])
                # display image
                imgFolder = os.path.join(utils.raw_path,
                                         params.traverses[traverse],
                                         'stereo/left')
                imgPath = os.path.join(imgFolder,
                                       str(query_tstamps[indQ]) + '.png')
            # import camera model to undistort images
            camera = camera_model.CameraModel(
                utils.raw_path + 'camera-models/', imgFolder)
            img = image.load_image(imgPath, model=camera)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].imshow(img)
            if j == nCols - 1:
                axs[i, j].yaxis.set_label_position("right")
                axs[i, j].set_ylabel(traverse, fontsize=32)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(utils.figures_path + '/sample_images.jpg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SeqSLAM on trials")
    parser.add_argument('-r', '--reference', type=str, default='Overcast',
                        help="reference traverse used as the map")
    parser.add_argument('-q', '--queries', nargs='+', type=str,
                        default=['Rain', 'Dusk', 'Night'],
                        help="Names of query traverses to localize"
                        "against reference map e.g. Overcast, Night,"
                        "Dusk etc. Input 'all' instead to process all"
                        "traverses. See src/params.py for full list.")
    args = parser.parse_args()
    main(args)
