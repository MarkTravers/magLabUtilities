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
#   change functions to input a dependent SignalThread and output a Signal
#
# \revised
#   Mark Travers 4/22/2020      - Original construction
#   Stephen Gedney 4/27/2020    - Added Linear function

import numpy as np
from typing import Tuple, Callable
from magLabUtilities.signalutilities.signals import SignalThread, Signal
from magLabUtilities.exceptions.exceptions import SignalValueError

class FunctionSequence:
    def __init__(self, t0=np.float64(0.0)):
        self.functionList = []
        self.tTotal = t0

    def appendFunction(self, function:Callable[[SignalThread], Signal], functionT0:np.float64, functionT1:np.float64):
        self.functionList.append((function, functionT0, functionT1, functionT1-functionT0))
        self.tTotal += abs(functionT1 - functionT0)

    def evaluate(self, tThread:SignalThread):
        t = np.float64(0.0)
        if not tThread.isIncreasing:
            raise SignalValueError('tThread must be increasing.')

        signalList = []
        functionRegion = None
        for function in self.functionList:
            if function is self.functionList[-1]:
                functionRegion = np.where(np.logical_and(tThread.data >= t, tThread.data <= t+function[3]))
            else:
                functionRegion = np.where(np.logical_and(tThread.data >= t, tThread.data < t+function[3]))
            t += function[3]

            signalList.append(function(0)(tThread[functionRegion[0], functionRegion[-1]], tThread[functionRegion[0]]))

        return Signal.fromSignalSequence(signalList)

class Line:
    def __init__(self, x0:np.float64, x1:np.float64, t0:np.float64, t1:np.float64, enforceTBounds=False):
        self.x0 = x0
        self.x1 = x1
        self.t0 = t0
        self.t1 = t1
        self.enforceTBounds = enforceTBounds

    def evaluate(self, tThread:SignalThread, tOffset:np.float64) -> Signal:
        if self.enforceTBounds:
            if not (np.all(tThread.data - tOffset >= self.t0) and np.all(tThread.data - tOffset <= self.t1)):
                raise SignalValueError('Cannot evaluate function outside bounds.')
        if not tThread.isIncreasing:
            raise SignalValueError('tThread must be increasing.')

        return Signal.fromThreadPair((tThread.data - tOffset) * (self.x1-self.x0) / (self.t1-self.t0) - self.x0, tThread)

class SeriesApproximation:
    pass

# class FunctionGenerator:
#     def __init__(self):
#         self.signalList = []

#     def appendSignal(self, signalPiece):
#         if len(self.signalList) == 0:
#             self.signalList.append(signalPiece)
#         else:
#             self.signalList.append(signalPiece[1:])

#     def compileSignal(self):
#         return np.hstack(self.signalList)

#     def parabola(self, xMaxStep, xMinStep, x0, x1, power, numPoints):
#         def mapToParabola(x, exponent):
#             if (xMaxStep > xMinStep and xMinStep > x) or (xMaxStep < xMinStep and xMinStep < x):
#                 x = xMinStep + (xMinStep - x)
#                 return -(xMaxStep - xMinStep) * np.power((x-xMinStep)/(xMaxStep-xMinStep), exponent) + xMinStep
#             else:
#                 return (xMaxStep - xMinStep) * np.power((x-xMinStep)/(xMaxStep-xMinStep), exponent) + xMinStep

#         t0 = mapToParabola(x0, 1.0/power)
#         t1 = mapToParabola(x1, 1.0/power)
#         tArray = np.linspace(t0, t1, num=numPoints)
#         mapArrayToParabola = np.vectorize(mapToParabola)
#         xArray = mapArrayToParabola(tArray, power)
#         return xArray

#     def sinusoid(self, amplitude, thetaStep, angularPeriod, numCycles, phaseShift=0.0):
#         numPoints = (angularPeriod * numCycles) / thetaStep
#         tArray = np.linspace(0.0, numPoints*thetaStep, numPoints+1)
#         return amplitude * np.sin(tArray-phaseShift)

#     def linear(self, xStep, xStart, xStop):
#         numPoints = np.floor(np.absolute((xStop - xStart) / xStep))
#         npts = numPoints.astype(np.int)+1
#         xArray = np.linspace(xStart, xStop, npts)
#         return xArray

# if __name__ == '__main__':
#     # Create a signal generator
#     sigGen = FunctionGenerator()

#     # ################ Minor loop example with parabolic point distribution #################
#     # # Virgin curve
#     # sigGen.appendSignal(sigGen.parabola(xMaxStep=10000.0, xMinStep=0.0, x0=0.0, x1=10000.0, power=1.5, numPoints=101))
#     # # Positive reversal
#     # sigGen.appendSignal(sigGen.parabola(xMaxStep=10000.0, xMinStep=-820.0, x0=10000.0, x1=-10000.0, power=1.5, numPoints=201))
#     # # Negative reversal
#     # sigGen.appendSignal(sigGen.parabola(xMaxStep=-10000.0, xMinStep=820.0, x0=-10000.0, x1=10000.0, power=1.5, numPoints=201))

#     ################# Minor loop example with sinusoidal point distribution #################
#     # sigGen.appendSignal(sigGen.sinusoid(amplitude=10000.0, thetaStep=np.pi/100, angularPeriod=2.0*np.pi, numCycles=1.25, phaseShift=0.0)) 

#     # combine signals, excluding shared intermediary endpoints
#     signal = sigGen.compileSignal()

#     # calculate and display largest and smallest step sizes
#     stepSizes = np.abs(signal[:-1] - signal[1:])
#     print('Largest step size: %f' % np.amax(stepSizes))
#     print('Smallest step size: %f' % np.amin(stepSizes))
#     print('Mean step size: %f' % np.mean(stepSizes))
#     print('Standard deviation of step size: %f' % np.std(stepSizes))

#     # output to waveform file
#     with open('./waveform.txt', 'w') as waveformFile:
#         waveformFile.write('%s\n1.0\n' % signal.shape[0])
#         for i in range(signal.shape[0]):
#             waveformFile.write('%f\n' % signal[i])
