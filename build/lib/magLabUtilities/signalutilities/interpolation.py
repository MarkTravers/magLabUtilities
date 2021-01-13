#!python3

import numpy as np
from magLabUtilities.signalutilities.signals import SignalThread, Signal
from magLabUtilities.signalutilities.calculus import integralTrapQuadrature
# from internal.terminalInterface import printMsg
# from internal.exceptions import IncompatibleArgumentError, DataShapeError

def nearestPoint(interpSignal:Signal, tThread:SignalThread) -> Signal:
    tDistanceMatrix = np.vstack([interpSignal.dependentThread.data]*tThread.length)
    np.abs(tDistanceMatrix - np.transpose([tThread.data]), out=tDistanceMatrix)
    interpIndices = np.argmin(tDistanceMatrix, axis=1)

    independentThread = SignalThread(interpSignal.independentThread.data[interpIndices])
    dependentThread = SignalThread(interpSignal.dependentThread.data[interpIndices])

    return Signal.fromThreadPair(independentThread, dependentThread)

class Legendre:
    def __init__(self, interpRadius:int, legendreOrder:int):
        self.interpRadius = interpRadius
        self.legendreOrder = legendreOrder

    # Ouputs the value of the first 'order' orders of Legendre Polynomials evaluated at x
    @staticmethod
    def legendrePolynomial(x:float, order:int) -> np.ndarray:
        # Input validation
        if(order < 0):
            pass
            # raise IncompatibleArgumentError('Legendre polynomial cannot be order %d.' % order) \todo - make exception
        # Initialize first two legendre coefficients
        p = np.zeros((order + 1))
        p[0] = 1
        if(order < 1):
            return p
        p[1] = x

        # Calculate remaining legendre coefficients
        for i in range(2,order+1):
            p[i] = ((2.0 * i - 1.0)*x*p[i-1] - (i - 1.0)*p[i-2]) / i
        return p

    def interpolate(self, interpSignal:Signal, tThread:SignalThread) -> Signal:
        tDistanceMatrix = np.vstack([interpSignal.dependentThread.data]*tThread.length)
        np.abs(tDistanceMatrix - np.transpose([tThread.data]), out=tDistanceMatrix)
        interpIndices = np.argmin(tDistanceMatrix, axis=1)

        outputData = np.array(np.zeros_like(tThread.data))

        for i, interpIndex in enumerate(interpIndices):
            if interpIndex - self.interpRadius < 0:
                i0 = 0
                i1 = 2 * self.interpRadius + 1
            elif interpIndex + self.interpRadius > interpSignal.dependentThread.data.size:
                i0 = interpSignal.dependentThread.data.size - 1 - 2*self.interpRadius
                i1 = interpSignal.dependentThread.data.size - 1
            else:
                i0 = interpIndex - self.interpRadius
                i1 = interpIndex + self.interpRadius

            # Find Legendre polynomial values at t-points from the local range of interpSignal
            interpT = interpSignal.dependentThread.data[i0:i1]
            psiTArray = (interpT - np.amin(interpT)) * 2.0 / np.ptp(interpT) - 1.0
            legValueArray = np.zeros((self.legendreOrder + 1, psiTArray.size))
            for t, psiT in enumerate(psiTArray):
                legValueArray[:,t] = Legendre.legendrePolynomial(psiT, self.legendreOrder)

            # Calculate the weighting coefficients, alpha
            interpX = interpSignal.independentThread.data[i0:i1]
            alphaArray = np.zeros((self.legendreOrder + 1, 1))
            for m in range(self.legendreOrder+1):
                # innerProduct = integralTrapQuadrature(SignalThread(legValueArray[m,:] * interpX), SignalThread(psiTArray)).independentThread.data[-1]
                innerProduct = integralTrapQuadrature(SignalThread(legValueArray[m,:] * interpX), SignalThread(interpT),c=0.0).data[-1]

                alphaArray[m] = (2.0 * m + 1.0) / (interpT[-1] - interpT[0]) * innerProduct

            # Find Legendre polynomial values at current t in tThread
            psiT = (tThread.data[i] - np.amin(interpT)) * 2.0 / np.ptp(interpT) - 1.0
            legValueArray = np.empty((self.legendreOrder + 1, 1))
            legValueArray[:,0] = Legendre.legendrePolynomial(psiT, self.legendreOrder)

            # Calculate f(t) at current t in tThread
            outputData[i] = np.sum(legValueArray * alphaArray)

        return Signal.fromThreadPair(SignalThread(outputData), tThread)

def legendreFilter(data:np.ndarray, integrationWindowSize:int, stepSize:int, legendreOrder:int, terminalMode='quiet') -> np.ndarray:
    # Setup and input validation
    data = np.array(data)
    if len(data.shape) > 1:
        pass
        # raise DataShapeError('Input array must be one-dimensional.') \todo - make an exception
    inputLength = data.shape[0]
    outputLength = int(np.ceil(inputLength / stepSize))
    outputData = np.zeros((outputLength), dtype=data.dtype)

    windowRadius = int(np.floor(integrationWindowSize / 2))
    psiTArray = np.linspace(-1.0, 1.0, num=2*windowRadius + 1)
    legValueArray = np.empty((legendreOrder + 1, psiTArray.size))
    for t in range(psiTArray.size):
        legValueArray[:,t] = Legendre.legendrePolynomial(psiTArray[t], legendreOrder)

    # Interpolate data
    for outputIndex, inputIndex in enumerate(range(0, inputLength, stepSize)):
        # printMsg('Interpolating output point: %d/%d' % (outputIndex, outputLength), terminalMode)

        # Set integration window limits and enforce the window size at each end of the data
        if inputIndex < windowRadius:
            t0 = 0
            t1 = 2 * windowRadius
        elif inputIndex >= inputLength - windowRadius:
            t0 = inputLength - 2 * windowRadius - 1
            t1 = inputLength - 1
        else:
            t0 = inputIndex - windowRadius
            t1 = inputIndex + windowRadius

        sumTerms = legValueArray * data[t0:t1+1]
        sumTerms[:,0] = sumTerms[:,0] * 0.5
        sumTerms[:,-1] = sumTerms[:,-1] * 0.5
        innerProduct = np.sum(sumTerms, axis=1)

        # alpha[:,outputIndex] = innerProduct * ((2*np.arange(legendreOrder+1) + 1) / (t1 - t0))

        psiT = (2 * inputIndex - (t0 + t1)) / (t1 - t0)

        outputData[outputIndex] = np.sum(Legendre.legendrePolynomial(psiT, legendreOrder) * innerProduct * ((2*np.arange(legendreOrder+1) + 1) / (t1 - t0)))

    return outputData


