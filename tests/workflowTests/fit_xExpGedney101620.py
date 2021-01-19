#!python3

import numpy as np
from magLabUtilities.signalutilities.signals import SignalThread, Signal, SignalBundle
from magLabUtilities.signalutilities.hysteresis import HysteresisSignalBundle
from magLabUtilities.signalutilities.interpolation import Legendre

class Cost:
    def __init__(self, refFP:str):
        self.refBundle = self.importLoopSignal(refFP)

    def importLoopSignal(self, fp) -> HysteresisSignalBundle:
        # Import two-column .txt data into signal threads
        raw = np.genfromtxt(fp, np.float64)
        prIndex = np.argmax(raw[0:int(raw.shape[0]/2), 0])
        nrIndex = np.argmin(raw[:,0])
        hVirginRawSignal = Signal.fromSingleThread(SignalThread(raw[0:prIndex,0]), 'arcLength', tStart=0.0, totalArcLength=1.0)
        mVirginRawSignal = Signal.fromSingleThread(SignalThread(raw[0:prIndex,1]), 'arcLength', tStart=0.0, totalArcLength=1.0)
        hprRawSignal = Signal.fromSingleThread(SignalThread(raw[prIndex:nrIndex,0]), 'arcLength', tStart=1.0, totalArcLength=3.0)
        mprRawSignal = Signal.fromSingleThread(SignalThread(raw[prIndex:nrIndex,1]), 'arcLength', tStart=1.0, totalArcLength=3.0)
        hnrRawSignal = Signal.fromSingleThread(SignalThread(raw[nrIndex:,0]), 'arcLength', tStart=3.0, totalArcLength=5.0)
        mnrRawSignal = Signal.fromSingleThread(SignalThread(raw[nrIndex:,1]), 'arcLength', tStart=3.0, totalArcLength=5.0)

        # Filter and up-sample data
        dataFilter = Legendre(20, 3)
        tVirginThread = np.linspace(0.0, 1.0, 10000)
        hVirginSignal = dataFilter.interpolate(hVirginRawSignal, tVirginThread)
        mVirginSignal = dataFilter.interpolate(mVirginRawSignal, tVirginThread)
        tPRSignal = np.linspace(1.0, 3.0, 20000)
        hPRSignal = dataFilter.interpolate(hprRawSignal, tPRSignal)
        mPRSignal = dataFilter.interpolate(mprRawSignal, tPRSignal)
        tNRSignal = np.linspace(3.0, 5.0, 20000)
        hNRSignal = dataFilter.interpolate(hnrRawSignal, tNRSignal)
        mNRSignal = dataFilter.interpolate(mnrRawSignal, tNRSignal)

        bundle = HysteresisSignalBundle()
        bundle.addSignal('H', hSignal)
        bundle.addSignal('M', mSignal)
        return bundle

if __name__=='__main__':
    # xInit, mSat, mNuc, hCoercive, hAnh, hCoop
    parameterList = [
        {   'name':'xInit',
            'initialValue':69.0,
            'stepSize':3,
            'testGridLocalIndices':[0]
            # 'testGridLocalIndices':[-1,0,1]
        },
        {   'name':'mSat',
            'initialValue':1.66e6,
            'stepSize':0.01e6,
            'testGridLocalIndices':[0]
            # 'testGridLocalIndices':[-1,0,1]
        },
        {   'name':'mNuc',
            'initialValue':1.5221e6,
            'stepSize':0.03e6,
            # 'testGridLocalIndices':[0]
            'testGridLocalIndices':[-1,0,1]
        },
        {   'name':'hCoercive',
            'initialValue':680.0,
            'stepSize':25.0,
            'testGridLocalIndices':[0]
            # 'testGridLocalIndices':[-1,0,1]
        },
        {
            'name':'hAnh',
            'initialValue':4300.0,
            'stepSize':50.0,
            # 'testGridLocalIndices':[0]
            'testGridLocalIndices':[-1,0,1]
        },
        {
            'name':'hCoop',
            'initialValue':660.0,
            'stepSize':50.0,
            # 'testGridLocalIndices':[0]
            'testGridLocalIndices':[-1,0,1]
        }
    ]