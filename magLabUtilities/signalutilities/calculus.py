#!python3

import numpy as np
from typing import List
from magLabUtilities.signalutilities.signals import SignalThread, Signal
# from internal.terminalInterface import printMsg

def differentialDerivative(fNum, fDenom, windowRadius, discontinuousPoints, terminalMode='verbose'):
    dF = []
    for index in range(len(fNum)):
        windowIndices = [None] * 2
        for radius in range(windowRadius + 1):
            if index + radius in discontinuousPoints or index - radius in discontinuousPoints:
                if radius == 0:
                    if index == 0:
                        windowIndices[0] = index
                        windowIndices[1] = index + 1
                    else:
                        windowIndices[0] = index - 1
                        windowIndices[1] = index
                else:
                    windowIndices[0] = index - radius
                    windowIndices[1] = index + radius
        if windowIndices == [None] * 2:
            windowIndices[0] = index - windowRadius
            windowIndices[1] = index + windowRadius
        dF.append((fNum[windowIndices[1]] - fNum[windowIndices[0]]) / (fDenom[windowIndices[1]] - fDenom[windowIndices[0]]))
        # printMsg('Index: %d    F: %f    Left Index: %d    F(left): %f    Right Index: %d    F(right):    %f    dF: %f' % (index, fNum[index], windowIndices[0], fNum[windowIndices[0]], windowIndices[1], fNum[windowIndices[1]], dF[index]), terminalMode)
    return dF

def finiteDiffDerivative(fNum:np.ndarray, fDenom:np.ndarray, windowRadius:np.int, discontinuousPoints:List[np.int], differenceMode:str) -> np.ndarray:
    fNum = np.array(fNum, dtype=np.float32)
    fDenom = np.array(fDenom, dtype=np.float32)
    discontinuousPoints = discontinuousPoints + [0, fNum.size-1]
    dF = []
    for index in range(len(fNum)):
        windowIndices = [None] * 2
        for radius in range(windowRadius + 1):
            if index + radius in discontinuousPoints or index - radius in discontinuousPoints:
                if radius == 0:
                    if index == 0:
                        windowIndices[0] = index
                        windowIndices[1] = index + 1
                    else:
                        windowIndices[0] = index - 1
                        windowIndices[1] = index
                else:
                    windowIndices[0] = index - radius
                    windowIndices[1] = index + radius
        if windowIndices == [None] * 2:
            windowIndices[0] = index - windowRadius
            windowIndices[1] = index + windowRadius
        if differenceMode == 'centralDifference':
            dF.append((fNum[windowIndices[1]] - fNum[windowIndices[0]]) / (fDenom[windowIndices[1]] - fDenom[windowIndices[0]]))
        if differenceMode == 'backwardsDifference':
            if index == 0:
                dF.append((fNum[1] - fNum[windowIndices[0]]) / (fDenom[1] - fDenom[windowIndices[0]]))
            else:
                dF.append((fNum[index] - fNum[windowIndices[0]]) / (fDenom[index] - fDenom[windowIndices[0]]))
        # print ('Index: %d    F: %e    Left Index: %d    F(left): %e    Right Index: %d    F(right):    %e    dF: %e' % (index, fNum[index], windowIndices[0], fNum[windowIndices[0]], windowIndices[1], fNum[windowIndices[1]], dF[index]))
    return np.array(dF, dtype=np.float32)

def integralIndexQuadrature(independentThread:np.ndarray, dependentThread:np.ndarray, c:np.float64=0.0) -> SignalThread:
    dT = np.ediff1d(dependentThread, to_begin=dependentThread[0])
    return SignalThread(np.cumsum(independentThread.data * dT) + c)

def integralTrapQuadrature(independentThread:SignalThread, dependentThread:SignalThread, c:np.float64=0.0) -> SignalThread:
    dT = np.ediff1d(dependentThread.data)
    dX = np.ediff1d(independentThread.data)

    outputArray = np.array(np.zeros_like(independentThread.data), dtype=np.float64)
    outputArray[1:] = dT * independentThread.data[:-1] + 0.5 * dT * dX

    return SignalThread(np.cumsum(outputArray) + c)

# integral(data, a, b, quadratureMode=['dataPoints', 'quadrature'], params)

# derivative(xData, yData, differentiationMode=['legendre', 'slopeControlledWindow'], singularityMode=['legendre', 'adaptiveWindow'])
