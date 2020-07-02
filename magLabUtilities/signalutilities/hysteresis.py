#!python3

# \name
#   hysteresis curve utilities
#
# \description
#   Module for common manipulations of hysteresis curves
#
# \notes
#
# \todo
#
# \revised
#   Mark Travers 6/16/2020      - Original construction

import numpy as np
from magLabUtilities.signalutilities.signals import SignalThread, Signal, SignalBundle
from magLabUtilities.exceptions.exceptions import SignalTypeError

class HysteresisSignalBundle(SignalBundle):
    def __init__(self, signalBundle:SignalBundle=None):
        super().__init__()
        if isinstance(signalBundle, SignalBundle):
            self.signals = signalBundle.signals
        elif signalBundle is not None:
            raise SignalTypeError('Cannot cast %s to HysteresisSignalBundle.' % str(type(signalBundle)))

    @property
    def mhComplete(self):
        if 'M' in self.signals.keys() and 'H' in self.signals.keys():
            return True
        else:
            return False

    @property
    def xmComplete(self):
        if 'X' in self.signals.keys() and 'M' in self.signals.keys():
            return True
        else:
            return False

    @property
    def xmhComplete(self):
        if 'X' in self.signals.keys() and 'M' in self.signals.keys() and 'H' in self.signals.keys():
            return True
        else:
            return False

class XExpQA:
    def __init__(self, xInit:np.float64, hCoercive:np.float64, mSat:np.float64, hCoop:np.float64, hAnh:np.float64, xcPow:np.float64, mRev:np.float64, virginMTolerance:np.float64):
        self.xInit = np.float64(xInit)
        self.hCoercive = np.float64(hCoercive)
        self.mSat = np.float64(mSat)
        self.hCoop = np.float64(hCoop)
        self.hAnh = np.float64(hAnh)
        self.xcPow = np.float64(xcPow)
        self.mRev = np.float64(mRev)
        self.virginMTolerance = np.float64(virginMTolerance)

    def evaluate(self, mSignal:Signal) -> HysteresisSignalBundle:
        if abs(self.mRev) <= self.virginMTolerance:
            absDM = np.abs(mSignal.independentThread.data)
            hCTerm = self.hCoercive
        else:
            absDM = np.abs(mSignal.independentThread.data - self.mRev)
            hCTerm = self.hCoop

        xr = 1.0 - np.power(np.abs(mSignal.independentThread.data / self.mSat), 2)
        xc = 1.0 - np.power(np.abs(mSignal.independentThread.data / self.mSat), self.xcPow)
        xRev = self.xInit * xr
        mSatMM = np.power(self.mSat, 2) - np.power(mSignal.independentThread.data, 2)

        exponent = absDM / (self.xInit * (hCTerm + (self.hAnh*self.mSat*absDM)/(xc*mSatMM)))
        xSignal = Signal.fromThreadPair(SignalThread(xRev * np.exp(exponent)), mSignal.dependentThread)

        xOfMBundle = HysteresisSignalBundle()
        xOfMBundle.addSignal('M', mSignal)
        xOfMBundle.addSignal('X', xSignal)

        return xOfMBundle

class XExpGedney060820:
    def __init__(self, xInit:np.float64, hCoercive:np.float64, mSat:np.float64, hCoop:np.float64, hAnh:np.float64, xcPow:np.float64, mRev:np.float64, virginMTolerance:np.float64):
        self.xInit = np.float64(xInit)
        self.hCoercive = np.float64(hCoercive)
        self.mSat = np.float64(mSat)
        self.hCoop = np.float64(hCoop)
        self.hAnh = np.float64(hAnh)
        self.xcPow = np.float64(xcPow)
        self.mRev = np.float64(mRev)
        self.virginMTolerance = np.float64(virginMTolerance)

    def evaluate(self, mSignal:Signal) -> HysteresisSignalBundle:
        if abs(self.mRev) <= self.virginMTolerance:
            hCTerm = self.hCoercive / 2.0
        else:
            hCTerm = self.hCoop

        absDM = np.abs(mSignal.independentThread.data - self.mRev)
        xr = 1.0 - np.power(np.abs(mSignal.independentThread.data / self.mSat), 2)
        xc = 1.0 - np.power(np.abs(mSignal.independentThread.data / self.mSat), self.xcPow)
        xRev = self.xInit * xr

        mSatMM = mSignal.independentThread.data
        gtIndices = np.where(mSignal.independentThread.data - self.mRev > 0)
        mSatMM[gtIndices] = self.mSat - mSignal.independentThread.data[gtIndices]
        ltIndices = np.where(mSignal.independentThread.data - self.mRev <= 0)
        mSatMM[ltIndices] = self.mSat + mSignal.independentThread.data[ltIndices]

        exponent = absDM / (self.xInit * (hCTerm/xc + (self.hAnh*absDM)/mSatMM))
        xSignal = Signal.fromThreadPair(SignalThread(xRev * np.exp(exponent)), mSignal.dependentThread)

        xOfMBundle = HysteresisSignalBundle()
        xOfMBundle.addSignal('M', mSignal)
        xOfMBundle.addSignal('X', xSignal)

        return xOfMBundle
