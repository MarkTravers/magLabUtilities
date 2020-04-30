#!python3

import numpy as np
from internal.terminalInterface import printMsg
from internal.exceptions import IncompatibleArgumentError, DataShapeError

def legendre(data, integrationWindowSize, stepSize, legendreOrder, terminalMode='quiet'):
    # Setup and input validation
    data = np.array(data)
    if len(data.shape) > 1:
        raise DataShapeError('Input array must be one-dimensional.')
    inputLength = data.shape[0]
    outputLength = int(np.ceil(inputLength / stepSize))
    outputData = np.zeros((outputLength), dtype=data.dtype)

    windowRadius = int(np.floor(integrationWindowSize / 2))
    psiTArray = np.linspace(-1.0, 1.0, num=2*windowRadius + 1)
    legValueArray = np.empty((legendreOrder + 1, psiTArray.size))
    for t in range(psiTArray.size):
        legValueArray[:,t] = legendrePolynomial(psiTArray[t], legendreOrder)

    # Interpolate data
    for outputIndex, inputIndex in enumerate(range(0, inputLength, stepSize)):
        printMsg('Interpolating output point: %d/%d' % (outputIndex, outputLength), terminalMode)

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

        outputData[outputIndex] = np.sum(legendrePolynomial(psiT, legendreOrder) * innerProduct * ((2*np.arange(legendreOrder+1) + 1) / (t1 - t0)))

    return outputData

# Ouputs the value of the first 'order' orders of Legendre Polynomials evaluated at x
def legendrePolynomial(x, order):
    # Input validation
    if(order < 0):
        raise IncompatibleArgumentError('Legendre polynomial cannot be order %d.' % order)
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
