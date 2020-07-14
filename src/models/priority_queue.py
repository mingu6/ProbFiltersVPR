from node import Node

class QueueElement:
    def __init__(self, quId, refId, idvCost):
        self.quId = quId
        self.refId = refId
        self.idvCost = idvCost


class Graph:
    def __init__(self):
        # store parents
        self._pred = {}
        self._pred[-1] = {-1: Node(-1, -1, 0.0)}
        # store accumulated costs for all nodes
        self._accCosts = {}
        self._accCosts[-1] = {-1: 0.0}
        # initialize queue with source node
        source = QueueElement(-1, -1, 0.0)
        self.PQ = [source]

    def empty(self):
        return len(self.PQ) == 0

    def update(self):
        """
        Sorts the priority queue based on _accCosts
        """
        self.PQ = sorted(self.PQ, key=lambda node:
                         self._accCosts[node.quId][node.refId])

    def pop(self):
        """
        Pops queue element with most priority and updates graph in 
        accordance.
        """
        popEl = self.PQ.pop(0)
        #self._pred[popEl.quId].pop(popEl.refId)
        #if not self._pred[popEl.quId]:
        #    self._pred.pop(popEl.quId)
        #self._accCosts[popEl.quId].pop(popEl.refId)
        #if not self._accCosts[popEl.quId]:
        #    self._accCosts.pop(popEl.quId)
        return popEl

    def push_back(self, el):
        self.PQ.append(el)
        self.update()
