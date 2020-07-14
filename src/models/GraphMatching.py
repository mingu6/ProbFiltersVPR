import os
import argparse
import pickle
import time
from queue import PriorityQueue
from copy import deepcopy

import numpy as np
from tqdm import trange, tqdm

from src import utils, geometry
from node import Node, NodeSet
from priority_queue import Graph, QueueElement


class GraphMatching:
    def __init__(self, map_poses, map_descriptors, qSize,
                 expansionRate, fan_out):
        self.map_poses = map_poses
        self.map_descriptors = map_descriptors
        self._expandedRecently = NodeSet()
        self._graph = Graph()
        self._currentBestHyp = Node(-1, -1, 0.0)
        # parameters
        self._qSize = qSize
        self.expansionRate = expansionRate
        self._fan_out = fan_out

    def localize(self, query_descriptors):
        for qu in range(self._qSize):
            print(qu)
            self.processImage(qu, query_descriptors[qu])
        # if level qSize - 1 attained, success
        if self._currentBestHyp.quId == self._qSize - 1:
            return self.map_poses[self._currentBestHyp.refId], True
        else:
            return self.map_poses[self._currentBestHyp.refId], False

    def processImage(self, qu, query_descriptor):
        row_reached = False
        self._expandedRecently.clear()
        while (not self._graph.empty()) and (not row_reached):
            # check if node is worth expanding
            expandedEl = self._graph.pop()
            expanded_row = expandedEl.quId
            expanded_col = expandedEl.refId
            if not self.node_worth_expanding(expanded_row, expanded_col):
                continue
            # if worth expanding, retrieve children
            children = self.getSuccessors(expanded_row, expanded_col,
                                          query_descriptor)
            if children:
                # upgrade graph and queue
                self.updateGraph(expandedEl, children)
                self.updateSearch(children)
            else:
                print("[ERROR][OnlineLocalizer] Expanded node {}, {}"
                      "has no children. Do not know what to do! Exit"
                      .format(expanded_row, expanded_col))
                exit()
            self._expandedRecently.insert(children)
            if expanded_row == qu - 1:
                row_reached = True

    def node_worth_expanding(self, quId, refId):
        if quId == -1 and refId == -1:
            # source node always worth expanding
            return True
        if quId == self._currentBestHyp.quId and \
                refId == self._currentBestHyp.refId:
            return True
        row_dist = self._currentBestHyp.quId - quId
        if row_dist < 0:
            print("[ERROR][INTERNAL][GraphMatching] Trying to expand "\
                "a node further in future {} than current best "\
                  "hypothesis hypothesis {}".format(
                    quId, self._currentBestHyp.quId))
            exit()
        mean_cost = self.compute_average_path_cost()
        acc_cost = self._graph._accCosts[quId][refId]
        potential_cost = acc_cost + row_dist * mean_cost * \
            self.expansionRate
        if potential_cost < \
                self._graph._accCosts[self._currentBestHyp.quId]\
                    [self._currentBestHyp.refId]:
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
            pred = self._graph._pred[pred.quId][pred.refId]
        mean_cost = mean_cost / elInPath
        return mean_cost

    def getSuccessors(self, quId, refId, query_descriptor):
        if quId == -1 and refId == -1:
            # for source node, expand all references
            cost = np.sqrt(2 - 2 * self.map_descriptors @
                                query_descriptor)
            successors = [Node(quId + 1, i, cost[i]) for i in
                               range(len(cost))]
            return successors
        if quId < 0 or refId < 0:
            print("[ERROR][getSuccessors] Invalid Ids {} {}".format(
                quId, refId))
            exit()
        # get fan out successors
        left_ref = max(refId - self._fan_out, 0)
        right_ref = min(refId + self._fan_out, len(self.map_descriptors)
                        - 1)

        cost = np.sqrt(
            2 - 2 * self.map_descriptors[left_ref:right_ref+1] @
                            query_descriptor)
        successors = [Node(quId + 1, left_ref + i, cost[i]) for i in 
                        range(len(cost))]
        return successors

    def updateGraph(self, parent, successors):
        # for every successor
        # check if the child was visited before (i.e. parents exist)
        # if so, check if proposed accumulated cost is smaller than
        # implied by parents, else set predecessor for child
        for node in successors:
            if self.predExists(node):
                pa_accCost = \
                    self._graph._accCosts[parent.quId][parent.refId]
                prev_accCost = \
                    self._graph._accCosts[node.quId][node.refId]
                poss_accCost = node.idvCost + pa_accCost
                # child was visited before
                if poss_accCost < prev_accCost:
                    print("Email Olga! See original C++ code")
                    self._graph._pred[node.quId][node.refId] = \
                        Node(parent.quId, parent.refId, parent.idvCost)
                    self._graph._accCosts[node.quId][node.refId] = \
                        poss_accCost
                    self._graph.update()
            else:
                # new successors
                if node.quId not in self._graph._pred:
                    self._graph._pred[node.quId] = {node.refId: Node(
                            parent.quId, parent.refId, parent.idvCost
                        )
                    }
                else:
                    self._graph._pred[node.quId][node.refId] = Node(
                            parent.quId, parent.refId, parent.idvCost
                        )
                parent_cost = \
                    self._graph._accCosts[parent.quId][parent.refId]
                if node.quId not in self._graph._accCosts:
                    self._graph._accCosts[node.quId] = {
                        node.refId: node.idvCost + parent_cost
                    }
                else:
                    poss_accCost = node.idvCost + parent_cost
                    self._graph._accCosts[node.quId][node.refId] = \
                        poss_accCost
                succEl = QueueElement(node.quId, node.refId,
                                      node.idvCost)
                self._graph.push_back(succEl)

    def predExists(self, node):
        if node.quId not in self._graph._pred:
            return False
        return node.refId in self._graph._pred[node.quId]

    def updateSearch(self, successors):
        possibleHyp = self.get_prominent_successor(successors)
        if possibleHyp.quId > self._currentBestHyp.quId:
            self._currentBestHyp = possibleHyp
        elif possibleHyp.quId == self._currentBestHyp.quId:
            accCost_current = self._graph._accCosts\
                [self._currentBestHyp.quId][self._currentBestHyp.refId]
            accCost_poss = self._graph._accCosts\
                [possibleHyp.quId][possibleHyp.refId]
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
    ref_poses, ref_descriptors, _ = utils.import_reference_map(
        args.reference_traverse)
    # localize all selected query traverses
    pbar = tqdm(args.query_traverses)
    for traverse in pbar:
        pbar.set_description(traverse)
        # savepath
        save_path = os.path.join(utils.results_path, traverse)
        # load query data
        query_poses, _, _, query_descriptors, _ = \
            utils.import_query_traverse(traverse)
        # regular traverse with VO
        pbar = tqdm(args.descriptors, leave=True)
        for desc in pbar:
            pbar.set_description(desc)
            # one folder per descriptor
            save_path1 = os.path.join(save_path, desc)
            if not os.path.exists(save_path1):
                os.makedirs(save_path1)
            model = GraphMatching(ref_poses, ref_descriptors[desc],
                                  args.seq_len, args.exp_rate,
                                  args.fan_out)
            proposals, scores, times, query_gt = \
                utils.localize_traverses_matching(model,
                                                  query_poses,
                                                  query_descriptors
                                                  [desc]
                                                  [:, :args.seq_len, :],
                                                  desc='Graph',
                                                  idx=args.seq_len)
            # testing!!!
            proposals_c = geometry.combine(proposals)
            query_gt_c = geometry.combine(query_gt)
            dists = np.linalg.norm((proposals_c / query_gt_c).t(), axis=1)
            print(dists[:100])

            utils.save_obj(save_path1 + '/Graph.pickle',
                           model='Graph', query_gt=query_gt,
                           proposals=proposals, scores=scores,
                           times=times, L=args.seq_len)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run graph matching")
    parser.add_argument('-r', '--reference-traverse',
                        type=str, default='Overcast',
                        help="reference traverse used as the map")
    parser.add_argument('-q', '--query-traverses', nargs='+',
                        type=str, default=['Sun', 'Dusk', 'Night'],
                        help="Names of query traverses to localize"
                        "against reference map e.g. Overcast, Night,"
                        "Dusk etc. Input 'all' instead to process all"
                        "traverses. See src/params.py for full list.")
    parser.add_argument('-d', '--descriptors', nargs='+', type=str,
                        default=['NetVLAD', 'DenseVLAD'],
                        help='descriptor types to run experiments on.')
    parser.add_argument('-L', '--seq-len', type=int, default=10,
                        help="Sequence length for sequence matching")
    parser.add_argument('-e', '--exp-rate', type=float, default=0.5,
                        help="expansion rate parameter")
    parser.add_argument('-W', '--fan-out', type=int, default=10,
                        help="Fan out parameter for expansion")
    args = parser.parse_args()

    main(args)
