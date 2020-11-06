#!python3

import numpy as np
from magLabUtilities.signalutilities.signals import Signal, SignalBundle, SignalThread
from magLabUtilities.signalutilities.hysteresis import HysteresisSignalBundle
from magLabUtilities.signalutilities.interpolation import Legendre, nearestPoint
from magLabUtilities.optimizerutilities.costFunctions import rmsNdNorm
from magLabUtilities.optimizerutilities.parameterSpaces import GridNode
from magLabUtilities.optimizerutilities.gradientDescent import GradientDescent
from magLabUtilities.uiutilities.plotting.hysteresis import MofHPlotter

def circle(tThread:SignalThread, xRad:np.float64=1.0, yRad:np.float64=1.0, xCen:np.float64=0.0, yCen:np.float64=0.0) -> HysteresisSignalBundle:
    xSignal = Signal.fromThreadPair(SignalThread(np.cos(tThread.data) * xRad + xCen), tThread)
    ySignal = Signal.fromThreadPair(SignalThread(np.sin(tThread.data) * yRad + yCen), tThread)
    circleBundle = HysteresisSignalBundle()
    circleBundle.addSignal('H', xSignal)
    circleBundle.addSignal('M', ySignal)

    return circleBundle

class Cost:
    def __init__(self, refData:SignalBundle):
        self.refData = refData

    def evaluate(self, gridNode:GridNode):
        # Collect ref bundle
        gridNode.refData = self.refData

        # Generate test bundle
        testTThread = SignalThread(np.linspace(0.0, gridNode.coordList[4], 137))
        gridNode.testData = circle(testTThread, xRad=gridNode.coordList[0], yRad=gridNode.coordList[1], xCen=gridNode.coordList[2], yCen=gridNode.coordList[3])

        # Sample ref and test bundles for re-parameterization
        refArray = gridNode.refData.sample(gridNode.refData.signals['H'].dependentThread, [('H', nearestPoint), ('M', nearestPoint)])
        testArray = gridNode.testData.sample(gridNode.testData.signals['H'].dependentThread, [('H', nearestPoint), ('M', nearestPoint)])
        # Re-parameterize ref and test bundles
        refArray = SignalBundle.arcLengthND(refArray, totalArcLength=1.0, normalizeAxes=True)
        # Sample ref and test bundles for error calculation
        sampleTThread = SignalThread(np.linspace(0.0, 1.0, 10.0))
        refArray = gridNode.refData.sample(sampleTThread, [('H', nearestPoint), ('M', nearestPoint)])
        testArray = gridNode.testData.sample(sampleTThread, [('H', nearestPoint), ('M', nearestPoint)])

        # Calculate error
        rmsNdNorm(refArray, testArray, normalizeDataByDimRange=True)

class Plotter:
    def __init__(self):
        self.plotter = MofHPlotter()

    def plotGridNode(self, gridNode:GridNode):
        self.plotter.addPlot(gridNode.testData, plotName='')

    def plotBundle(self, signalBundle:SignalBundle, plotName=''):
        self.plotter.addPlot(signalBundle, plotName)

if __name__ == '__main__':
    # Initialize plotter
    plotter = Plotter()

    # Generate "data" circle
    tThread = SignalThread(np.linspace(0.0, 2.0*np.pi, 100))
    dataBundle = circle(tThread, xRad=1.0, yRad=1.0, xCen=0.0, yCen=0.0)
    plotter.plotBundle(dataBundle, 'Data')

    # Set up optimizer
    parameterList = [
        {   'name':'xRad',
            'initialValue':69.0,
            'stepSize':3,
            'testGridLocalIndices':[0]
            # 'testGridLocalIndices':[-1,0,1]
        },
        {   'name':'yRad',
            'initialValue':69.0,
            'stepSize':3,
            'testGridLocalIndices':[0]
            # 'testGridLocalIndices':[-1,0,1]
        },
        {   'name':'xCen',
            'initialValue':69.0,
            'stepSize':3,
            'testGridLocalIndices':[0]
            # 'testGridLocalIndices':[-1,0,1]
        },
        {   'name':'yCen',
            'initialValue':69.0,
            'stepSize':3,
            'testGridLocalIndices':[0]
            # 'testGridLocalIndices':[-1,0,1]
        },
        {   'name':'tTotal',
            'initialValue':69.0,
            'stepSize':3,
            'testGridLocalIndices':[0]
            # 'testGridLocalIndices':[-1,0,1]
        }
    ]

    cost = Cost(dataBundle)
    tuner = GradientDescent(parameterList, cost.evaluate, plotter.plotGridNode)
    tuner.tune(numIterations=np.infty, maxThreads=8)

    # Loop
    #   Collect neighbor nodes
    #   Generate "model" circle
    #   Evaluate loss function on each node
    #   Choose lowest loss
    #   Plot

    print('done.')
