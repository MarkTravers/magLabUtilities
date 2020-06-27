#!python3

import numpy as np
from magLabUtilities.optimizers.costFunctions import rmsNdNorm
from magLabUtilities.signalutilities.signals import SignalThread, Signal, SignalBundle

if __name__=='__main__':
    tThread = SignalThread(np.array([0,1,2], dtype=np.float64))

    refM = SignalThread(np.array([0,1,2], dtype=np.float64))
    refM = Signal.fromThreadPair(refM, tThread)
    refH = SignalThread(np.array([0,1,2], dtype=np.float64))
    refH = Signal.fromThreadPair(refH, tThread)

    testM = SignalThread(np.array([0,1,2], dtype=np.float64))
    testM = Signal.fromThreadPair(testM, tThread)
    testH = SignalThread(np.array([0,2,4], dtype=np.float64))
    testH = Signal.fromThreadPair(testH, tThread)

    refBundle = SignalBundle()
    refBundle.addSignal('M', refM)
    refBundle.addSignal('H', refH)
    refMatrix = refBundle.sample(tThread, [('M','nearestPoint'), ('H','nearestPoint')])

    testBundle = SignalBundle()
    testBundle.addSignal('M', testM)
    testBundle.addSignal('H', testH)
    testMatrix = testBundle.sample(tThread, [('M','nearestPoint'), ('H','nearestPoint')])

    tWeight = np.array([[0,1,2],[1,1,1]], dtype=np.float64)
    cost = rmsNdNorm(refMatrix, testMatrix, tWeight=tWeight, normalizeDataByDimRange=False)

    print(cost)