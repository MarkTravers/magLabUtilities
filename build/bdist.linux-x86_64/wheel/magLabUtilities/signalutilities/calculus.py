#!python3

import numpy as np
from internal.terminalInterface import printMsg

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
        printMsg('Index: %d    F: %f    Left Index: %d    F(left): %f    Right Index: %d    F(right):    %f    dF: %f' % (index, fNum[index], windowIndices[0], fNum[windowIndices[0]], windowIndices[1], fNum[windowIndices[1]], dF[index]), terminalMode)
    return dF

def finiteDiffDerivative(fNum, fDenom, windowRadius, discontinuousPoints, differenceMode):
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

# integral(data, a, b, quadratureMode=['dataPoints', 'quadrature'], params)

# derivative(xData, yData, differentiationMode=['legendre', 'slopeControlledWindow'], singularityMode=['legendre', 'adaptiveWindow'])
