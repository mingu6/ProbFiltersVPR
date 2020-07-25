import os
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.rcParams["pdf.fonttype"] = 42
# matplotlib.rcParams["ps.fonttype"] = 42
import pickle

from tabulate import tabulate

from src import utils, geometry
from src.params import descriptors
from src.evaluation import evaluate


colors = {
    "MCL": "limegreen",
    "MCL RTK motion": "green",
    "Topological": "dodgerblue",
    "Seq Match": "orangered",
    "Single": "orange",
    "Graph": "red",
}

linestyle = {
    "MCL": "solid",
    "MCL RTK motion": "solid",
    "Topological": "solid",
    "Seq Match": "dashed",
    "Single": "dashed",
    "Graph": "dotted",
}


def main(args):
    nQueries = len(args.query_traverses)
    nThres = len(args.t_thres)
    results_path_0 = os.path.join(
        utils.results_path, args.query_traverses[0], args.descriptors[0]
    )
    models = [model[:-7] for model in sorted(os.listdir(results_path_0))]
    nModels = len(models)

    for desc in args.descriptors:
        # initialize PR curves
        fig, axs = plt.subplots(nQueries, nThres, figsize=(6 * nThres, 3.5 * nQueries))
        # initialize results arrays
        max_recall_results = np.empty((nQueries, nThres, nModels))
        AUC_results = max_recall_results.copy()
        compute_results = max_recall_results.copy()
        nsteps_results = max_recall_results.copy()
        for i in range(nQueries):
            results_path = os.path.join(
                utils.results_path, args.query_traverses[i], desc
            )
            for j, (t_thres, R_thres) in enumerate(zip(args.t_thres, args.R_thres)):
                # initialize PR curve plots
                axs[i, j].set_aspect(0.5)
                if i == 0:
                    axs[i, j].set_title(
                        "{:.0f}m, {:.0f}deg".format(t_thres, R_thres), fontsize=20
                    )
                axs[i, j].set_xlabel("Recall", fontsize=14)
                axs[i, j].set_ylabel("Precision", fontsize=14)
                axs[i, j].set_xlim(0, 1.0)
                axs[i, j].set_ylim(0.0, 1.1)
                for k, fname in enumerate(sorted(os.listdir(results_path))):
                    with open(os.path.join(results_path, fname), "rb") as f:
                        results = pickle.load(f)
                    # extract results data from dict
                    model = results["model"]
                    query_poses = results["query_gt"]
                    proposals = results["proposals"]
                    scores = results["scores"]
                    times = np.asarray(results["times"])

                    # generate PR curve data using scores,
                    # metric error and thresholds
                    if model in ["Seq Match", "Single", "Topological", "Graph"]:
                        PR = evaluate.PRCurve(t_thres, R_thres * np.pi / 180, model)
                        PR.generate_curve(query_poses, proposals, scores)
                        axs[i, j].plot(
                            PR.recalls,
                            PR.precisions,
                            label=model,
                            linestyle=linestyle[model],
                            color=colors[model],
                            linewidth=3,
                        )
                    else:
                        # For particle filter models, we run multiple trials
                        # Generate PR curves over all trials
                        PRs = []
                        for l, (p, s) in enumerate(zip(proposals, scores)):
                            # evaluate curves, using same set of scores to
                            # general all PR curve points
                            PR = evaluate.PRCurve(t_thres, R_thres * np.pi / 180, model)
                            PR.generate_curve(query_poses, p, s)
                            if l > 0:
                                PR.interpolate_at_recalls(PRs[0].recalls)
                            PRs.append(PR)
                        PR_mean, PR_min, PR_max = evaluate.max_min_avg_PR(PRs)
                        axs[i, j].plot(
                            PR_mean.recalls,
                            PR_mean.precisions,
                            label=model,
                            linestyle=linestyle[model],
                            color=colors[model],
                            linewidth=3,
                        )
                        axs[i, j].fill_between(
                            PR_mean.recalls,
                            PR_min.precisions,
                            PR_max.precisions,
                            facecolor=colors[model],
                            interpolate=True,
                            alpha=0.3,
                        )
                    # generate table results
                    if model in ["Seq Match", "Single"]:
                        # find indices at score threshold for precision
                        # level and retrieve compute time and num steps
                        score_at, recall_at = PR.score_precision_level(
                            args.precision_level
                        )
                        localized = PR._localize_indices(scores, score_at)
                        total_compute = times[localized != -1]
                        nsteps = np.ones_like(total_compute)
                        if model == "Seq Match":
                            nsteps *= results["L"]
                    else:
                        # table stuff
                        if model in ["Topological", "Graph"]:
                            score_at, recall_at = PR.score_precision_level(
                                args.precision_level
                            )
                            step_localized = PR._localize_indices(scores, score_at)
                            compute_at_localized = np.squeeze(
                                np.take_along_axis(
                                    np.asarray(times), step_localized[:, np.newaxis], 1
                                )
                            )
                        else:
                            score_at, recall_at = PR_mean.score_precision_level(
                                args.precision_level
                            )
                            step_localized = PR_mean._localize_indices(
                                scores[0], score_at
                            )
                            compute_at_localized = np.squeeze(
                                np.take_along_axis(
                                    np.asarray(times[0]),
                                    step_localized[:, np.newaxis],
                                    1,
                                )
                            )
                        total_compute = compute_at_localized[step_localized != -1]
                        nsteps = step_localized[step_localized != -1] + 1
                    if model in ["Single", "Seq Match", "Topological", "Graph"]:
                        AUC_results[i, j, k] = PR.auc()
                    else:
                        AUC_results[i, j, k] = PR_mean.auc()
                    max_recall_results[i, j, k] = recall_at
                    compute_results[i, j, k] = np.mean(total_compute * 1000 / nsteps)
                    nsteps_results[i, j, k] = np.mean(nsteps)
                    # side headings for traverse name
                    if j == 0:
                        axs[i, j].text(
                            -0.22,
                            0.40,
                            args.query_traverses[i],
                            fontsize=20,
                            rotation=90,
                        )
        # plot PR curves and save
        axs[-1, -1].legend(loc="lower right", fontsize=12)
        fig.suptitle(desc, fontsize=24)
        fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
        plt.savefig(utils.figures_path + "/PR_{}.png".format(desc), bbox_inches="tight")
        # create tables using results
        table_maxrecall = []
        table_AUC = []
        table_compute = []
        table_nsteps = []
        for i in range(nModels):
            # create table rows
            table_maxrecall.append(
                [models[i], *max_recall_results[..., i].reshape(-1).tolist()]
            )
            table_AUC.append([models[i], *AUC_results[..., i].reshape(-1).tolist()])
            table_compute.append(
                [models[i], *compute_results[..., i].reshape(-1).tolist()]
            )
            table_nsteps.append(
                [models[i], *nsteps_results[..., i].reshape(-1).tolist()]
            )

        headers = ["Models"]
        for traverse in args.query_traverses:
            for t_thres, R_thres in zip(args.t_thres, args.R_thres):
                headers.append("{} {}m, {}deg".format(traverse, t_thres, R_thres))
        fname_out = os.path.join(utils.figures_path, "table_{}.txt".format(desc))
        print("Descriptor: {}\n".format(desc), file=open(fname_out, "w"))
        print(
            "Max recall at {}\n\n".format(args.precision_level),
            tabulate(table_maxrecall, floatfmt=".3f", headers=headers),
            file=open(fname_out, "a"),
        )
        print(
            "\nAUC\n\n",
            tabulate(table_AUC, floatfmt=".3f", headers=headers),
            file=open(fname_out, "a"),
        )
        print(
            "\nCompute time per iteration\n\n",
            tabulate(table_compute, floatfmt=".1f", headers=headers),
            file=open(fname_out, "a"),
        )
        print(
            "\nNumber of steps until localized\n\n",
            tabulate(table_nsteps, floatfmt=".1f", headers=headers),
            file=open(fname_out, "a"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sequencematching on trials")
    parser.add_argument(
        "-q",
        "--query-traverses",
        nargs="+",
        type=str,
        default=["Rain", "Dusk", "Night"],
        help=(
            "Names of query traverses to plot results"
            "for e.g. Overcast, Night, Dusk etc."
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
        "-t",
        "--t-thres",
        type=float,
        nargs="+",
        default=[3, 5],
        help=(
            "maximum error for translation distance (m)                         for a"
            " correct localization"
        ),
    )
    parser.add_argument(
        "-R",
        "--R-thres",
        type=float,
        nargs="+",
        default=[15, 30],
        help=(
            "maximum error for angle (degrees) for a                         correct"
            " localization"
        ),
    )
    parser.add_argument(
        "-p",
        "--precision-level",
        type=float,
        default=0.99,
        help="Precision level to evaluate tableelements at",
    )
    args = parser.parse_args()

    main(args)
