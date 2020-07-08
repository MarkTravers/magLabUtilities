#!python3

import numpy as np
from magLabUtilities.signalutilities.signals import Signal
from magLabUtilities.exceptions.exceptions import CostValueError

def rmsNdNorm(refMatrix:np.ndarray, testMatrix:np.ndarray, tWeightMatrix:np.ndarray=None, normalizeDataByDimRange=True) -> np.float64:
    if (refMatrix.shape != testMatrix.shape) or (not np.all(refMatrix[0,:] == testMatrix[0,:])):
        raise CostValueError('refMatrix and testMatrix must be sampled identically.')

    if tWeightMatrix is None:
        tWeightMatrix = np.vstack((refMatrix[0,:], np.ones((refMatrix.shape[1]), dtype=np.float64)))
    elif (tWeightMatrix.shape[0] != 2) or (not np.all(refMatrix[0,:] == tWeightMatrix[0,:])):
        raise CostValueError('tWeightMatrix must be sampled at the same times as refMatrix and testMatrix.')

    if normalizeDataByDimRange:
        # dimRanges = np.transpose([np.ptp(refMatrix, axis=1) + np.ptp(testMatrix, axis=1)]) * 0.5
        dimRanges = np.transpose([np.ptp(refMatrix, axis=1)])
        np.divide(refMatrix[1:,:], dimRanges[1:], out=refMatrix[1:,:])
        np.divide(refMatrix[2:,:], dimRanges[2:], out=refMatrix[2:,:])
        np.divide(testMatrix[1:,:], dimRanges[1:], out=testMatrix[1:,:])
        np.divide(testMatrix[2:,:], dimRanges[2:], out=testMatrix[2:,:])

    refTestDifferences = np.abs(refMatrix[1:,:] - testMatrix[1:,:])
    tWeightMatrixedDistances = np.multiply(np.linalg.norm(refTestDifferences, axis=0), tWeightMatrix[1,:])
    return np.sqrt(np.sum(tWeightMatrixedDistances) / tWeightMatrixedDistances.size, dtype=np.float64)
