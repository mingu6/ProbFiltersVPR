class Node:
    def __init__(self, quId, refId, idvCost):
        self.quId = quId
        self.refId = refId
        self.idvCost = idvCost

    def set(self, quId, refId, idvCost):
        self.quId = quId
        self.refID = refId
        self.idvCost = idvCost

    def print(self):
        print(
            "[INFO][Node] qu: {}; ref {}; cost: {:2.5f}".format(
                self.qId, self.refId, self.idvCost
            )
        )


class NodeSet:
    def __init__(self):
        self._nodes = {}

    def clear(self):
        self._nodes = {}

    def insert(self, v):
        for el in v:
            if el.quId in self._nodes:
                if el.refId not in self._nodes[el.quId]:
                    self._nodes[el.quId][el.refId] = el
            else:
                self._nodes[el.quId] = {}
                self._nodes[el.quId][el.refId] = el

    def toVector(self):
        nodesVec = []
        for quIds in self._nodes.values():
            for node in quIds.values():
                nodesVec.append(node)
        return nodesVec
