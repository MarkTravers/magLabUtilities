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

def evaluateCost(gridNode:GridNode):
    # Evaluate model

if __name__=='__main__':
    # Generate "data" circle
    tThread = SignalThread(np.linspace(0.0, 2.0*np.pi, 100))
    dataBundle = circle(tThread, xRad=1.0, yRad=1.0, xCen=0.0, yCen=0.0)

    # Set up optimizer
    plotter = MofHPlotter()
    plotter.addPlot(dataBundle, 'Data')


    # Loop
    #   Collect neighbor nodes
    #   Generate "model" circle
    #   Evaluate loss function on each node
    #   Choose lowest loss
    #   Plot

    print('done.')
