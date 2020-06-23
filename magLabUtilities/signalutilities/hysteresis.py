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

class XExpQA:
    def __init__(self, xInit:np.float64, hCoercive:np.float64, mSat:np.float64, hCoop:np.float64, hAnh:np.float64, xcPow:np.float64, mRev:np.float64):
        self.xInit = np.float64(xInit)
        self.hCoercive = np.float64(hCoercive)
        self.mSat = np.float64(mSat)
        self.hCoop = np.float64(hCoop)
        self.hAnh = np.float64(hAnh)
        self.xcPow = np.float64(xcPow)
        self.mRev = np.float64(mRev)

    def evaluate(self, mThread:SignalThread):
        if self.mRev == 0.0:
            mMMr = np.abs(mThread.data)
            hCTerm = self.hCoercive
        else:
            mMMr = np.abs(mThread.data - self.mRev)
            hCTerm = self.hCoop

        xr = 1.0 - np.power(mThread.data / self.mSat, 2)
        xc = 1.0 - np.power(mThread.data / self.mSat, self.xcPow)
        xRev = self.xInit * xr
        mSatMM = np.power(self.mSat, 2) - np.power(mThread.data, 2)

        exponent = mMMr / (self.xInit * (hCTerm + (self.hAnh*self.mSat*mMMr)/(xc*mSatMM)))
        return Signal.fromThreadPair(SignalThread(xRev * np.exp(exponent)), mThread)
