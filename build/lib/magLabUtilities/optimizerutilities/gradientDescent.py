#!python3

import numpy as np
from pathos.threading import ThreadPool
from typing import Callable, List, Dict, Union
from magLabUtilities.signalutilities.signals import SignalThread, Signal, SignalBundle
from magLabUtilities.optimizerutilities.parameterSpaces import TestGrid, GridNode

class GradientDescent:
    def __init__(self, parameterList:List[Dict[str,Union[str, float, List[int]]]], costFunction:Callable[[GridNode], None], gradientStepFunction:Callable[[GridNode],None]):
        self.parameterList = parameterList
        self.costFunction = costFunction
        self.gradientStepFunction = gradientStepFunction

        self.testGrid = TestGrid(parameterList)

    def tune(self, numIterations:int=np.infty, maxThreads=None):
        nodeIndices = [0] * len(self.parameterList)
        lowestLossNode = None
        iterationNum = 0
        oldLowestLoss = np.infty

        while iterationNum < numIterations:
            iterationNum += 1
            testGridNodeList = self.testGrid.getTestGridNodes(nodeIndices)

            costFunctionQueue = []
            for testGridNode in testGridNodeList:
                if testGridNode.loss == None:
                    costFunctionQueue.append(testGridNode)

            with ThreadPool(nodes=maxThreads) as workerPool:
                testGridNodeList = workerPool.map(self.costFunction, costFunctionQueue)

            loss = []
            for testGridNode in testGridNodeList:
                loss.append(testGridNode.loss)

            lowestLossNode = testGridNodeList[np.argmin(np.array(loss))]
            self.gradientStepFunction(lowestLossNode)
            if oldLowestLoss <= lowestLossNode.loss:
                break
            else:
                oldLowestLoss = lowestLossNode.loss
                nodeIndices = lowestLossNode.indexList
