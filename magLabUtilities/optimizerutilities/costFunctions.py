#!python3

import numpy as np
from magLabUtilities.signalutilities.signals import Signal
from magLabUtilities.exceptions.exceptions import CostValueError

def rmsNdNorm(refBundle:np.ndarray, testBundle:np.ndarray, tWeight:np.ndarray=None, normalizeDataByDimRange=True) -> np.float64:
    if (refBundle.shape != testBundle.shape) or (not np.all(refBundle[0,:] == testBundle[0,:])):
        raise CostValueError('refBundle and testBundle must be sampled identically.')

    if tWeight is None:
        tWeight = np.vstack((refBundle[0,:], np.ones((refBundle.shape[1]), dtype=np.float64)))
    elif (tWeight.shape[0] != 2) or (not np.all(refBundle[0,:] == tWeight[0,:])):
        raise CostValueError('tWeight must be sampled at the same times as refBundle and testBundle.')

    if normalizeDataByDimRange:
        dimRanges = np.transpose([np.ptp(refBundle, axis=1) + np.ptp(testBundle, axis=1)]) * 0.5
        np.divide(refBundle[1:,:], dimRanges[1:], out=refBundle[1:,:])
        np.divide(testBundle[1:,:], dimRanges[1:], out=testBundle[1:,:])

    refTestDifferences = np.abs(refBundle[1:,:] - testBundle[1:,:])
    tWeightedDistances = np.multiply(np.linalg.norm(refTestDifferences, axis=0), tWeight[1,:])
    return np.sqrt(np.sum(tWeightedDistances) / tWeightedDistances.size, dtype=np.float64)
