import os
import numpy as np
import argparse
import pickle
import time

from tqdm import trange, tqdm
import matplotlib.pyplot as plt

from src import geometry, utils
from src.params import descriptors


class TopologicalFilter:
    def __init__(
        self, map_poses, map_descriptors, delta, window_lower=-2, window_upper=10
    ):
        # store map data
        self.map_poses = map_poses
        self.map_descriptors = map_descriptors
        # initialize hidden states and obs lhood parameters
        self.delta = delta
        self.lambda1 = 0.0
        self.belief = None
        # parameters for prior transition matrix
        self.window_lower = window_lower
        self.window_upper = window_upper
        self.window_size = int((window_upper - window_lower) / 2)
        self.transition = np.ones(window_upper - window_lower)

    def initialize_model(self, descriptor):
        dists = np.sqrt(2 - 2 * np.dot(self.map_descriptors, descriptor))
        descriptor_quantiles = np.quantile(dists, [0.025, 0.975])
        self.lambda1 = np.log(self.delta) / (
            descriptor_quantiles[1] - descriptor_quantiles[0]
        )
        self.belief = np.exp(-self.lambda1 * dists)
        self.belief /= self.belief.sum()

    def obs_lhood(self, descriptor):
        vsim = np.exp(
            -self.lambda1 * np.sqrt(2 - 2 * np.dot(self.map_descriptors, descriptor))
        )
        return vsim

    def update(self, descriptor):
        w_l = self.window_lower
        if w_l < 0:
            conv_ind_l, conv_ind_h = np.abs(w_l), len(self.belief) + np.abs(w_l)
            bel_ind_l, bel_ind_h = 0, len(self.belief)
        else:
            conv_ind_l, conv_ind_h = 0, len(self.belief) - w_l
            bel_ind_l, bel_ind_h = w_l, len(self.belief)
        # apply prior transition matrix
        self.belief[bel_ind_l:bel_ind_h] = np.convolve(self.belief, self.transition)[
            conv_ind_l:conv_ind_h
        ]
        if w_l > 0:
            self.belief[:w_l] = 0.0
        # observation likelihood update
        self.belief *= self.obs_lhood(descriptor)
        self.belief /= self.belief.sum()

    def localize(self):
        max_bel = np.argmax(self.belief)
        nhood_inds = np.arange(
            max(max_bel - 2 * self.window_size, 0),
            min(max_bel + 2 * self.window_size, len(self.belief) - 1),
        )
        score = np.sum(self.belief[nhood_inds])
        proposal = self.map_poses[
            int(np.rint(np.average(nhood_inds, weights=self.belief[nhood_inds])))
        ]
        return proposal, score


def main(args):
    # load reference data
    ref_poses, ref_descriptors, _ = utils.import_reference_map(args.reference_traverse)
    # localize all selected query traverses
    pbar = tqdm(args.query_traverses)
    for traverse in pbar:
        pbar.set_description(traverse)
        # savepath
        save_path = os.path.join(utils.results_path, traverse)
        # load query data
        query_poses, _, _, query_descriptors, _ = utils.import_query_traverse(traverse)
        # regular traverse with VO
        pbar = tqdm(args.descriptors, leave=False)
        for desc in pbar:
            pbar.set_description(desc)
            # one folder per descriptor
            save_path1 = os.path.join(save_path, desc)
            if not os.path.exists(save_path1):
                os.makedirs(save_path1)
            model = TopologicalFilter(
                ref_poses,
                ref_descriptors[desc],
                args.delta,
                window_lower=args.window_lower,
                window_upper=args.window_upper,
            )
            proposals, scores, times = utils.localize_traverses_filter(
                model, query_descriptors[desc], vo=None, desc="Topological"
            )
            utils.save_obj(
                save_path1 + "/Topological.pickle",
                model="Topological",
                query_gt=query_poses,
                proposals=proposals,
                scores=scores,
                times=times,
            )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run topologicalfilter on trials")
    parser.add_argument(
        "-r",
        "--reference-traverse",
        type=str,
        default="Overcast",
        help="reference traverse used as the map",
    )
    parser.add_argument(
        "-q",
        "--query-traverses",
        nargs="+",
        type=str,
        default=["Rain", "Dusk", "Night"],
        help=(
            "Names of query traverses to localize"
            "against reference map e.g. Overcast, Night,"
            "Dusk etc. Input 'all' instead to process all"
            "traverses. See src/params.py for full list."
        ),
    )
    parser.add_argument(
        "-d",
        "--descriptors",
        nargs="+",
        type=str,
        default=descriptors,
        help="descriptor types to run experiments on.",
    )
    parser.add_argument(
        "-D",
        "--delta",
        type=float,
        default=5,
        help=(
            "multiple used for calibrating sensor"
            "update rate parameter. Assumes 6sigma change in"
            "image difference causes M times sensor update"
        ),
    )
    parser.add_argument(
        "-wl",
        "--window-lower",
        type=int,
        default=-2,
        help="minimum state transition in transitionmatrix",
    )
    parser.add_argument(
        "-wu",
        "--window-upper",
        type=int,
        default=10,
        help="maximum state transition in transitionmatrix",
    )
    args = parser.parse_args()
    main(args)
