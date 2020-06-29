#!python3

# \name
#   Function Generator module
#
# \description
#   Generates combinations of canonical waveforms
#
# \notes
#   parabola(self, xMaxStep, xMinStep, x0, x1, power, numPoints)
#       where: 
#           xMaxStep  - x-point with the largest step size (e.g. at saturation)
#           xMinStep  - x-point with the smallest step size (e.g. at coercive field)
#           power     - exponent on the parabola (fractional powers are acceptable)
#           x0        - beginning of the sweep
#           x1        - end of the sweep
#           numPoints - is the number of points in the sweep (inclusive)
#
#   sinusoid(self, amplitude, thetaStep, angularPeriod, numCycles, phaseShift=0.0)
#       where:
#           amplitude       - Amplitude of the sine wave
#           thetaStep       - step size of theta
#           angularPeriod   - angular period of sine wave (2pi is one cylce)
#           numCycles       - number of cycles the sine wave goes through
#           (phaseShift)    - shifts the sine wave (default is 0.0)
#
# \todo
#
# \revised
#   Mark Travers 4/22/2020      - Original construction
#   Stephen Gedney 4/27/2020    - Added Linear function
#   Mark Travers 6/16/2020      - Implemented function generators with Signals. Vectorized Parabola

import numpy as np
from typing import Tuple, Callable
from magLabUtilities.signalutilities.signals import SignalThread, Signal
from magLabUtilities.exceptions.exceptions import SignalValueError

class Line:
    def __init__(self, x0:np.float64, x1:np.float64, t0:np.float64, t1:np.float64, enforceTBounds=True):
        self.x0 = x0
        self.x1 = x1
        self.t0 = t0
        self.t1 = t1
        self.enforceTBounds = enforceTBounds

        self.slope = (self.x1-self.x0) / (self.t1-self.t0)
        self.xIntercept = self.x0 - self.slope * self.t0

    def evaluate(self, tThread:SignalThread) -> Signal:
        if self.enforceTBounds:
            if not (np.all(tThread.data >= self.t0) and np.all(tThread.data <= self.t1)):
                raise SignalValueError('Cannot evaluate function outside bounds.')
        if not tThread.isIncreasing:
            raise SignalValueError('tThread must be increasing.')

        return Signal.fromThreadPair(SignalThread(tThread.data * self.slope + self.xIntercept), tThread)

class Parabola:
    def __init__(self, tMaxDx:np.float64, tVertex:np.float64, xMaxDx:np.float64, xVertex:np.float64, power:np.float64, enforceTBounds=True):
        self.tMaxDx = np.float64(tMaxDx)
        self.tVertex = np.float64(tVertex)
        self.xMaxDx = np.float64(xMaxDx)
        self.xVertex = np.float64(xVertex)
        self.power = np.float64(power)
        self.enforceTBounds = enforceTBounds

    def evaluate(self, tThread:SignalThread) -> Signal:
        if not tThread.isIncreasing:
            raise SignalValueError('tThread must be increasing.')

        if self.tVertex > self.tMaxDx:
            inRegion = np.where(tThread.data >= self.tVertex)
            outRegion = np.where(tThread.data < self.tVertex)

            inThread = -(self.xMaxDx-self.xVertex) * np.power((-tThread.data[inRegion]+self.tVertex)/(self.tMaxDx-self.tVertex), self.power) + self.tVertex
            outThread = (self.xMaxDx-self.xVertex) * np.power((tThread.data[outRegion]-self.tVertex)/(self.tMaxDx-self.tVertex), self.power) + self.tVertex

            xThread = np.hstack((outThread, inThread))

        elif self.tVertex < self.tMaxDx:
            inRegion = np.where(tThread.data <= self.tVertex)
            outRegion = np.where(tThread.data > self.tVertex)

            inThread = -(self.xMaxDx-self.xVertex) * np.power((tThread.data[inRegion]-self.tVertex)/(self.tMaxDx-self.tVertex), self.power) + self.tVertex
            outThread = (self.xMaxDx-self.xVertex) * np.power((-tThread.data[outRegion]+self.tVertex)/(self.tMaxDx-self.tVertex), self.power) + self.tVertex

            xThread = np.hstack((inThread, outThread))

        return Signal.fromThreadPair(SignalThread(xThread), tThread)

class Sinusoid:
    def __init__(self, amplitude:np.float64, angularPeriod:np.float64, phaseShift:np.float64):
        self.amplitude = amplitude
        self.angularPeriod = angularPeriod
        self.phaseShift = phaseShift

    def evaluate(self, tThread:SignalThread) -> Signal:
        return SignalThread(self.amplitude * np.sin(tThread.data - self.phaseShift))

class SeriesApproximation:
    pass
