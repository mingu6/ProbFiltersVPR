import os
import argparse
import pickle
import time
from queue import PriorityQueue
from copy import deepcopy

import numpy as np
from tqdm import trange, tqdm

from src import utils, geometry
from src.params import descriptors
from node import Node, NodeSet
from priority_queue import SortedCollection


class GraphMatching:
    def __init__(self, map_poses, map_descriptors, qSize, expansionRate, fan_out):
        self.map_poses = map_poses
        self.map_descriptors = map_descriptors
        self._expandedRecently = NodeSet()
        self._currentBestHyp = Node(-1, -1, 0.0)
        self._cost_matrix = None
        self._pred = {-1: {-1: Node(-1, -1, 0.0)}}  # store parents
        self._accCosts = {-1: {-1: 0.0}}  # store accumulated costs
        self._key = lambda node: self._accCosts[node.quId][node.refId]
        self._PQ = SortedCollection(
            [Node(-1, -1, 0.0)], self._key
        )  # source node only at start
        # parameters
        self._qSize = qSize
        self.expansionRate = expansionRate
        self._fan_out = fan_out

    def reset(self):
        self._expandedRecently = NodeSet()
        self._currentBestHyp = Node(-1, -1, 0.0)
        self._cost_matrix = None
        self._pred = {-1: {-1: Node(-1, -1, 0.0)}}  # store parents
        self._accCosts = {-1: {-1: 0.0}}  # store accumulated costs
        self._PQ = SortedCollection(
            [Node(-1, -1, 0.0)], self._key
        )  # source node only at start

    def localize(self, query_descriptors):
        # store proposals, scores and times
        proposals = []
        scores = np.empty(self._qSize)
        times = np.empty(self._qSize)

        start = time.time()
        # compute cost matrix
        self._cost_matrix = np.sqrt(
            2 - 2 * query_descriptors @ self.map_descriptors.transpose()
        )
        time_costMatrix = (time.time() - start) / (self._qSize + 1)  # per iter
        for qu in range(self._qSize):
            start = time.time()
            score, proposal_ind = self.processImage(qu)
            iter_time = time.time() - start
            times[qu] = time_costMatrix * (qu + 1) + iter_time
            proposals.append(self.map_poses[proposal_ind])
            scores[qu] = score
        proposals = geometry.combine(proposals)
        return proposals, scores, times

    def empty(self):
        return len(self._PQ) == 0

    def processImage(self, qu):
        row_reached = False
        self._expandedRecently.clear()
        while (not self.empty()) and (not row_reached):
            # check if node is worth expanding
            expandedEl = self._PQ.pop()
            expanded_row = expandedEl.quId
            expanded_col = expandedEl.refId
            if not self.node_worth_expanding(expanded_row, expanded_col):
                continue
            # if worth expanding, retrieve children
            children = self.getSuccessors(expanded_row, expanded_col)
            if children:
                # upgrade graph and queue
                self.updateGraph(expandedEl, children)
                self.updateSearch(children)
            else:
                print(
                    "[ERROR][OnlineLocalizer] Expanded node {}, {}"
                    "has no children. Do not know what to do! Exit".format(
                        expanded_row, expanded_col
                    )
                )
                exit()
            self._expandedRecently.insert(children)
            # tqdm.write(
            # "Size of expansion {}".format(len(self._expandedRecently.toVector()))
            # )
            if expanded_row == qu - 1:
                row_reached = True
        # best current hypothesis is proposed location
        cost = self._currentBestHyp.idvCost
        ref_place = self._currentBestHyp.refId
        return cost, ref_place

    def node_worth_expanding(self, quId, refId):
        if quId == -1 and refId == -1:
            # source node always worth expanding
            return True
        if quId == self._currentBestHyp.quId and refId == self._currentBestHyp.refId:
            return True
        row_dist = self._currentBestHyp.quId - quId
        if row_dist < 0:
            # print(
            # "[ERROR][INTERNAL][GraphMatching] Trying to expand "
            # "a node further in future {} than current best "
            # "hypothesis hypothesis {}".format(quId, self._currentBestHyp.quId)
            # )
            exit()
        mean_cost = self.compute_average_path_cost()
        acc_cost = self._accCosts[quId][refId]
        potential_cost = acc_cost + row_dist * mean_cost * self.expansionRate
        if (
            potential_cost
            < self._accCosts[self._currentBestHyp.quId][self._currentBestHyp.refId]
        ):
            return True
        else:
            return False

    def compute_average_path_cost(self):
        mean_cost = 0
        source_reached = False
        elInPath = 0
        pred = deepcopy(self._currentBestHyp)

        while not source_reached:
            if pred.quId == -1 and pred.refId == -1:
                source_reached = True
                continue
            mean_cost += pred.idvCost
            elInPath += 1
            pred = self._pred[pred.quId][pred.refId]
        mean_cost = mean_cost / elInPath
        return mean_cost

    def getSuccessors(self, quId, refId):
        if quId == -1 and refId == -1:
            # for source node, expand all references
            cost = self._cost_matrix[0, :]
            successors = [Node(quId + 1, i, cost[i]) for i in range(len(cost))]
            return successors
        if quId < 0 or refId < 0:
            print("[ERROR][getSuccessors] Invalid Ids {} {}".format(quId, refId))
            exit()
        # get fan out successors
        left_ref = max(refId - self._fan_out, 0)
        right_ref = min(refId + self._fan_out, len(self.map_descriptors) - 1)
        cost = self._cost_matrix[quId + 1, left_ref : right_ref + 1]
        successors = [Node(quId + 1, left_ref + i, cost[i]) for i in range(len(cost))]
        return successors

    def updateGraph(self, parent, successors):
        # special update for source node, insert all successors unordered and
        # then sort
        if parent.quId == -1 and parent.refId == -1:
            self._pred[0] = {}
            self._accCosts[0] = {}
            for node in successors:
                self._pred[node.quId][node.refId] = Node(-1, -1, 0.0)
                self._accCosts[node.quId][node.refId] = node.idvCost
            self._PQ._setitem(successors)
        else:
            # for every successor
            # check if the child was visited before (i.e. parents exist)
            # if so, check if proposed accumulated cost is smaller than
            # implied by parents, else set predecessor for child
            for node in successors:
                if self.predExists(node):
                    pa_accCost = self._accCosts[parent.quId][parent.refId]
                    prev_accCost = self._accCosts[node.quId][node.refId]
                    poss_accCost = node.idvCost + pa_accCost
                    # child was visited before
                    if poss_accCost < prev_accCost:
                        tqdm.write("Email Olga! See original C++ code")
                        self._pred[node.quId][node.refId] = Node(
                            parent.quId, parent.refId, parent.idvCost
                        )
                        self._accCosts[node.quId][node.refId] = poss_accCost
                        exit()
                else:
                    # new successors
                    # for new successor, add link to parents in graph
                    if node.quId not in self._pred:
                        self._pred[node.quId] = {
                            node.refId: Node(parent.quId, parent.refId, parent.idvCost)
                        }
                    else:
                        self._pred[node.quId][node.refId] = Node(
                            parent.quId, parent.refId, parent.idvCost
                        )
                    parent_cost = self._accCosts[parent.quId][parent.refId]
                    poss_accCost = node.idvCost + parent_cost
                    if node.quId not in self._accCosts:
                        self._accCosts[node.quId] = {node.refId: poss_accCost}
                    else:
                        self._accCosts[node.quId][node.refId] = poss_accCost
                    self._PQ.insert(node)

    def predExists(self, node):
        if node.quId not in self._pred:
            return False
        return node.refId in self._pred[node.quId]

    def updateSearch(self, successors):
        possibleHyp = self.get_prominent_successor(successors)
        if possibleHyp.quId > self._currentBestHyp.quId:
            self._currentBestHyp = possibleHyp
        elif possibleHyp.quId == self._currentBestHyp.quId:
            accCost_current = self._accCosts[self._currentBestHyp.quId][
                self._currentBestHyp.refId
            ]
            accCost_poss = self._accCosts[possibleHyp.quId][possibleHyp.refId]
            if accCost_poss <= accCost_current:
                self._currentBestHyp = possibleHyp

    def get_prominent_successor(self, successors):
        min_cost = 1e10
        for node in successors:
            if node.idvCost < min_cost:
                min_cost = node.idvCost
                minCost_node = node
        return minCost_node


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
            L = len(query_descriptors[desc][0])
            model = GraphMatching(
                ref_poses, ref_descriptors[desc], L, args.exp_rate, args.fan_out
            )
            proposals, scores, times, query_gt = utils.localize_traverses_graph(
                model, query_poses, query_descriptors[desc], desc="Graph"
            )

            utils.save_obj(
                save_path1 + "/Graph.pickle",
                model="Graph",
                query_gt=query_gt,
                proposals=proposals,
                scores=scores,
                times=times,
            )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run graph matching")
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
        "-e", "--exp-rate", type=float, default=0.95, help="expansion rate parameter"
    )
    parser.add_argument(
        "-W", "--fan-out", type=int, default=10, help="Fan out parameter for expansion"
    )
    args = parser.parse_args()

    main(args)
