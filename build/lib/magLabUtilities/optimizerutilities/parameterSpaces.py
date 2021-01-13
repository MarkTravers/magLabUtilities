#!python3

import json
import numpy as np
from typing import Dict, List, Tuple, Union

# class ParameterSpace:
#     def __init__(self, parameterDefs:Dict[str, Dict[str, Union[float, List[float]]]]):
#         self.parameterDefs = parameterDefs

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
    def __init__(self, indexList:List[np.float64], parameterList):
        self._indexList = indexList
        self._coordList = []
        for index, parameter in enumerate(parameterList):
            self._coordList.append(parameter['initialValue'] + parameter['stepSize'] * float(self._indexList[index]))
        self._loss = None
        self._refPlotBundle = None
        self._testPlotBundle = None
        self.data = {}

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
