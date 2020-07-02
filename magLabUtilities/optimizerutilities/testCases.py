#!python3

import json
from typing import Dict, List, Tuple, Union

class TestGrid:
    def __init__(self, parameterList:List[Dict[str,Union[str, float, List[int]]]]):
        self.parameterList = parameterList

        self.testGridIndices = []
        self._globalNodeDict = {}

        self.buildLocalGridIndices([], 0)

    def getTestGridNodes(self, currentNodeIndices):
        testGridNodeList = []
        for testGridNode in self.testGridIndices:

            nodeIndexList = [testGridNode[i] + currentNodeIndices[i] for i in range(len(currentNodeIndices))]
            nodeIndexKey = json.dumps(nodeIndexList)
            if nodeIndexKey not in list(self._globalNodeDict.keys()):
                self._globalNodeDict[nodeIndexKey] = GridNode(nodeIndexList, self.parameterList)
            testGridNodeList.append(self._globalNodeDict[nodeIndexKey])
        return testGridNodeList

    def buildLocalGridIndices(self, indexList:List[int], parameterIndex:int) -> None:
        for localIndex in self.parameterList[parameterIndex]['testGridLocalIndices']:
            indexList.append(localIndex)
            if parameterIndex < len(self.parameterList) - 1:
                self.buildLocalGridIndices(indexList, parameterIndex + 1)
            else:
                self.testGridIndices.append(indexList)
            indexList = indexList[:parameterIndex]

    @property
    def globalNodeDict(self):
        return self._globalNodeDict

    @property
    def testGridNodeNum(self):
        return len(self.testGridIndices)

class GridNode:
    def __init__(self, indexList, parameterList):
        self._indexList = indexList
        self._coordList = []
        for index, parameter in enumerate(parameterList):
            self._coordList.append(parameter['initialValue'] + parameter['stepSize'] * float(self._indexList[index]))
        self._loss = None
        self._data = None

        # self.mp.out(['GridNode'], 'Created GridNode with %s and %s' % (self._indexList, self._coordList))

    @property
    def indexList(self):
        return(self._indexList)

    @property
    def coordList(self):
        return(self._coordList)

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, loss):
        self._loss = loss

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
