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
from magLabUtilities.signalutilities.calculus import integralTrapQuadrature
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

    @property
    def xRevmComplete(self):
        if 'Xrev' in self.signals.keys() and 'M' in self.signals.keys():
            return True
        else:
            return False

class XExpQAGedney061720:
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

class XExpQACarl060820:
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
            hCTerm = self.hCoercive / (2.0 * absDM)
        else:
            absDM = np.abs(mSignal.independentThread.data - self.mRev)
            hCTerm = self.hCoop / (2.0 * absDM)

        xr = 1.0 - np.power(np.abs(mSignal.independentThread.data / self.mSat), 2)
        xc = 1.0 - np.power(np.abs(mSignal.independentThread.data / self.mSat), self.xcPow)
        xRev = self.xInit * xr

        hAnhTerm = self.hAnh / (self.mSat * np.power(1.0 - np.power(np.abs(mSignal.independentThread.data / self.mSat), 2), 2))
        exponent = xc / (self.xInit * (hCTerm + hAnhTerm))
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

        mSatMM = np.array(np.zeros_like(mSignal.independentThread.data, np.float64))
        gtIndices = np.where(mSignal.independentThread.data - self.mRev >= 0)[0]
        mSatMM[gtIndices] = self.mSat - mSignal.independentThread.data[gtIndices]
        ltIndices = np.where(mSignal.independentThread.data - self.mRev < 0)
        mSatMM[ltIndices] = self.mSat + mSignal.independentThread.data[ltIndices]

        exponent = absDM / (self.xInit * (hCTerm/xc + (self.hAnh*absDM)/mSatMM))
        xSignal = Signal.fromThreadPair(SignalThread(xRev * np.exp(exponent)), mSignal.dependentThread)

        xOfMBundle = HysteresisSignalBundle()
        xOfMBundle.addSignal('M', mSignal)
        xOfMBundle.addSignal('X', xSignal)

        return xOfMBundle

class XExpMremGedney070720:
    def __init__(self, xInit:float, hCoercive:float, mRem:float, mNuc:float, mSat:float, hAnh:float, mRev:float, virginMTolerance:float):
        self.xInit = np.float64(xInit)
        self.hCoercive = np.float64(hCoercive)
        self.mRem = np.float64(mRem)
        self.mNuc = np.float64(mNuc)
        self.mSat = np.float64(mSat)
        self.hAnh = np.float64(hAnh)
        self.mRev = np.float64(mRev)
        self.virginMTolerance = np.float64(virginMTolerance)

    def evaluate(self, mSignal:Signal) -> HysteresisSignalBundle:
        # Detect if this is the virgin curve
        if abs(self.mRev) <= abs(self.virginMTolerance):
            absDM = np.abs(mSignal.independentThread.data)
            hCTerm = self.hCoercive / (2.0 * absDM)
        # Detect if this is a positive reversal
        elif self.mRev > abs(self.virginMTolerance):
            absDM = np.abs(mSignal.independentThread.data - self.mRev)

            aboveNucIndices = np.where(mSignal.independentThread.data < -abs(self.mNuc))
            betweenIndices = np.where(np.logical_and(mSignal.independentThread.data > -abs(self.mNuc), mSignal.independentThread.data < -abs(self.mRem)))
            belowRemIndices = np.where(mSignal.independentThread.data > -abs(self.mRem))
            hCoop = np.array(np.empty_like(mSignal.independentThread.data))
            hCoop[aboveNucIndices] = self.hCoercive
            hCoop[betweenIndices] = self.hCoercive * np.abs(mSignal.independentThread.data[betweenIndices] - abs(self.mRem)) / (abs(self.mNuc) - abs(self.mRem))
            hCoop[belowRemIndices] = 0.0

            hCTerm = hCoop / (4.0 * absDM)
        # Detect if this is a negative reversal
        elif self.mRev < -abs(self.virginMTolerance):
            absDM = np.abs(mSignal.independentThread.data - self.mRev)

            aboveNucIndices = np.where(mSignal.independentThread.data > abs(self.mNuc))
            betweenIndices = np.where(np.logical_and(mSignal.independentThread.data < abs(self.mNuc), mSignal.independentThread.data > abs(self.mRem)))
            belowRemIndices = np.where(mSignal.independentThread.data < abs(self.mRem))
            hCoop = np.array(np.empty_like(mSignal.independentThread.data))
            hCoop[aboveNucIndices] = self.hCoercive
            hCoop[betweenIndices] = self.hCoercive * np.abs(mSignal.independentThread.data[betweenIndices] + abs(self.mRem)) / (abs(self.mNuc) - abs(self.mRem))
            hCoop[belowRemIndices] = 0.0

            hCTerm = hCoop / (4.0 * absDM)
        
        xr = 1.0 - np.power(np.abs(mSignal.independentThread.data / self.mSat), 2)
        xRev = self.xInit * xr

        hAnhTerm = self.hAnh / (self.mSat * np.power(1.0 - np.power(np.abs(mSignal.independentThread.data / self.mSat), 2), 2))

        exponent = 1.0 / (self.xInit * (hCTerm + hAnhTerm))
        xSignal = Signal.fromThreadPair(SignalThread(xRev * np.exp(exponent)), mSignal.dependentThread)

        xOfMBundle = HysteresisSignalBundle()
        xOfMBundle.addSignal('M', mSignal)
        xOfMBundle.addSignal('X', xSignal)

        return xOfMBundle

# class XExpTravers071520:
#     def __init__(self, xInit, hRev, mRev, hCoercive, mCoercive, hRem, mRem, hAnh, mSat, excitationDirection):
#         self.xInit = np.float64(xInit)
#         self.hRev = np.float64(hRev)
#         self.mRev = np.float64(mRev)
#         self.hCoercive = np.float64(hCoercive)
#         self.mCoercive = np.float64(mCoercive)
#         self.hRem = np.float64(hRem)
#         self.mRem = np.float64(mRem)
#         self.hAnh = np.float64(hAnh)
#         self.mSat = np.float64(mSat)

#         self.excitationDirection = excitationDirection

#     def evaluate(self, mSignal:Signal) -> HysteresisSignalBundle:
#         xRev = self.xInit * (1.0 * np.square(mSignal.independentThread.data / self.mSat))

#         if self.excitationDirection == 'increasing':
#             revTerm = self.hRev / np.abs(mSignal.independentThread.data - self.mRev)
#             coerciveTerm = self.hCoercive / np.abs(mSignal.independentThread.data - self.mCoercive)
#             remTerm = self.hRem / np.abs(mSignal.independentThread.data - self.mRem)
#             anhTerm = self.hAnh / (self.mSat - np.abs())

# Does not function correctly
class XExpGedney071720:
    def __init__(self, xInit:float, hCoercive:float, hNuc:float, mNuc:float, mSat:float, hCoop:float, hAnh:float, hRev:float, mRev:float, curveRegion:str):
        self.xInit = np.float64(abs(xInit))
        self.hCoercive = np.float64(abs(hCoercive))
        self.hNuc = np.float64(abs(hNuc))
        self.mNuc = np.float64(abs(mNuc))
        self.mSat = np.float64(abs(mSat))
        self.hCoop = np.float64(abs(hCoop))
        self.hAnh = np.float64(abs(hAnh))

        self.hRev = np.float64(hRev)
        self.mRev = np.float64(mRev)
        self.curveRegion = curveRegion

    def evaluate(self, mSignal:Signal) -> HysteresisSignalBundle:
        xSignal = Signal.fromThreadPair(SignalThread(np.zeros_like(mSignal.independentThread.data)), mSignal.dependentThread)
        hSignal = Signal.fromThreadPair(SignalThread(np.zeros_like(mSignal.independentThread.data)), mSignal.dependentThread)

        mSignal.independentThread.data[0] = self.mRev

        xr = 1.0 - np.power(mSignal.independentThread.data / self.mSat, 2)
        xRev = self.xInit * xr
        absDM = np.abs(mSignal.independentThread.data - self.mRev)
        hAnhTerm = (self.hAnh * absDM) / (self.mSat * np.power(1.0 - np.power(np.abs(mSignal.independentThread.data / self.mSat), 2), 2))

        for i in range(xSignal.independentThread.length):
            if abs(mSignal.independentThread.data[i]) > self.mNuc:
                xSignal.independentThread.data[i] = xRev[i]
            else:
                if self.curveRegion == 'virgin':
                    hCTerm = self.hCoercive * 0.5
                elif self.curveRegion == 'reversal':
                    if self.hRev < -self.hNuc:
                        self.hRev = -self.hNuc
                    elif self.hRev > self.hNuc:
                        self.hRev = self.hNuc

                    x = self.hRev - mSignal.independentThread.data[i]

                    if abs(x) > abs(self.hRev):
                        x = 1.0
                    else:
                        x = abs(x) / abs(self.hRev)
                    hCTerm = (self.hCoop - self.hCoercive) * (x**2) + self.hCoercive

                exponent = absDM[i] / (self.xInit * (hCTerm + hAnhTerm))
                xSignal.independentThread.data[i] = xRev[i] * np.exp(exponent)[0]

            if i+1 < hSignal.independentThread.length:
                hSignal.independentThread.data[i+1] = hSignal.independentThread.data[i] + (mSignal.independentThread.data[i+1] - mSignal.independentThread.data[i]) / xSignal.independentThread.data[i]

        hysteresisBundle = HysteresisSignalBundle()
        hysteresisBundle.addSignal('M', mSignal)
        hysteresisBundle.addSignal('H', hSignal)
        hysteresisBundle.addSignal('X', xSignal)

        xRevBundle = HysteresisSignalBundle()
        xRevBundle.addSignal('M', mSignal)
        xRevBundle.addSignal('X', Signal.fromThreadPair(SignalThread(xRev), mSignal.dependentThread))

        return hysteresisBundle, xRevBundle

# Does not function correctly
class XExpGedneyAlt071620:
    def __init__(self, xInit:float, hCoercive:float, hCoop:float, mNuc:float, mSat:float, hAnh:float, mRev:float, curveRegion:str):
        self.xInit = np.float64(abs(xInit))
        self.hCoercive = np.float64(abs(hCoercive))
        self.hCoop = np.float(abs(hCoop))
        self.mNuc = np.float64(abs(mNuc))
        self.mSat = np.float64(abs(mSat))
        self.hAnh = np.float64(abs(hAnh))
        self.mRev = np.float64(mRev)
        self.curveRegion = curveRegion

    def evaluate(self, mSignal:Signal) -> HysteresisSignalBundle:
        if self.curveRegion == 'virgin':
            absDM = np.abs(mSignal.independentThread.data)
            hCTerm = self.hCoercive / 2.0
        else:
            absDM = np.abs(mSignal.independentThread.data - self.mRev)
            
            if self.mRev < -self.mNuc:
                self.mRev = -self.mNuc
            elif self.mRev > self.mNuc:
                self.mRev = self.mNuc

            belowRemIndices = np.where(np.abs(self.mRev - mSignal.independentThread.data) < abs(self.mRev))
            aboveRemIndices = np.where(np.abs(self.mRev - mSignal.independentThread.data) >= abs(self.mRev))

            hCTerm = np.array(np.empty_like(mSignal.independentThread.data))
            hCTerm[belowRemIndices] = (self.hCoop - self.hCoercive) * ((self.mRev - mSignal.independentThread.data[belowRemIndices]) / self.mNuc) + self.hCoercive
            hCTerm[aboveRemIndices] = (self.hCoop - self.hCoercive) * (self.mRev / self.mNuc) + self.hCoercive
        
        xr = 1.0 - np.power(np.abs(mSignal.independentThread.data / self.mSat), 2)
        xRev = self.xInit * xr

        hAnhTerm = (self.hAnh * absDM) / (self.mSat * np.power(1.0 - np.power(np.abs(mSignal.independentThread.data / self.mSat), 2), 2))

        exponent = absDM / (self.xInit * (hCTerm + hAnhTerm))
        xSignal = Signal.fromThreadPair(SignalThread(xRev * np.exp(exponent)), mSignal.dependentThread)

        xOfMBundle = HysteresisSignalBundle()
        xOfMBundle.addSignal('M', mSignal)
        xOfMBundle.addSignal('X', xSignal)

        return xOfMBundle

class XExpOfHGedney071720:
    def __init__(self, xInit:float, hCoercive:float, hNuc:float, mNuc:float, mSat:float, hCoop:float, hAnh:float):
        self.xInit = np.float64(abs(xInit))
        self.hCoercive = np.float64(abs(hCoercive))
        self.hNuc = np.float64(abs(hNuc))
        self.mNuc = np.float64(abs(mNuc))
        self.mSat = np.float64(abs(mSat))
        self.hCoop = np.float64(abs(hCoop))
        self.hAnh = np.float64(abs(hAnh))

    def evaluate(self, hSignal:Signal, hRev:float, mRev:float, curveRegion:str) -> HysteresisSignalBundle:
        

        hRev = np.float64(hRev)
        mRev = np.float64(mRev)
        curveRegion = curveRegion

        xSignal = Signal.fromThreadPair(SignalThread(np.zeros_like(hSignal.independentThread.data)), hSignal.dependentThread)
        mSignal = Signal.fromThreadPair(SignalThread(np.zeros_like(hSignal.independentThread.data)), hSignal.dependentThread)
        xRevSignal = Signal.fromThreadPair(SignalThread(np.zeros_like(hSignal.independentThread.data)), hSignal.dependentThread)

        mSignal.independentThread.data[0] = mRev

        if mRev > self.mNuc:
            mRev = self.mNuc
        elif mRev <= -self.mNuc:
            mRev = -self.mNuc

        if hRev < -self.hNuc:
            hRev = -self.hNuc
        elif hRev >= self.hNuc:
            hRev = self.hNuc

        for i in range(xSignal.independentThread.length):
            xr = 1.0 - np.power(mSignal.independentThread.data[i] / self.mSat, 2)
            xRevSignal.independentThread.data[i] = self.xInit * xr
            absDM = np.abs(mSignal.independentThread.data[i] - mRev)
            hAnhTerm = (self.hAnh * absDM) / (self.mSat * np.power(1.0 - np.power(np.abs(mSignal.independentThread.data[i] / self.mSat), 2), 2))

            if abs(mSignal.independentThread.data[i]) > self.mNuc:
                xSignal.independentThread.data[i] = xRevSignal.independentThread.data[i]
            else:
                if curveRegion == 'virgin':
                    hCTerm = self.hCoercive * 0.5
                elif curveRegion == 'reversal':
                    x = hRev - hSignal.independentThread.data[i]

                    if abs(x) > abs(hRev):
                        x = 1.0
                    else:
                        x = abs(x) / abs(hRev)
                    hCTerm = (self.hCoop - self.hCoercive) * (x**1) + self.hCoercive

                exponent = absDM / (self.xInit * (hCTerm + hAnhTerm))
                xSignal.independentThread.data[i] = xRevSignal.independentThread.data[i] * np.exp(exponent)

            if i+1 < hSignal.independentThread.length:
                mSignal.independentThread.data[i+1] = mSignal.independentThread.data[i] + (hSignal.independentThread.data[i+1] - hSignal.independentThread.data[i]) * xSignal.independentThread.data[i]

        hysteresisBundle = HysteresisSignalBundle()
        hysteresisBundle.addSignal('M', mSignal)
        hysteresisBundle.addSignal('H', hSignal)
        hysteresisBundle.addSignal('X', xSignal)
        hysteresisBundle.addSignal('Xrev', xRevSignal)

        return hysteresisBundle

class XExpGendey101620:
    def __init__(self, xInit:np.float64, mSat:np.float64, mNuc:np.float64, hCoercive:np.float64, hAnh:np.float64, hCoop:np.float64):
        self.xInit = xInit
        self.mSat = mSat
        self.mNuc = mNuc
        self.hCoercive = hCoercive
        self.hAnh = hAnh
        self.hCoop = hCoop

    def evaluate(self, mSignal:Signal, mRev:np.float64, hRev:np.float64, curveRegion:str):
        # Calculate intermediate expressions
        m = np.divide(mSignal.independentThread.data, self.mSat)
        xR = 1.0 - np.power(m, 2)
        dM = np.abs(np.subtract(mSignal.independentThread.data, mRev))

        # Setup constants which change for virgin and reversal regions in and out of nucleation
        if curveRegion == 'virgin':
            hCP = self.hCoercive * 0.5
        elif curveRegion == 'reversal':
            if abs(mRev) > self.mNuc:
                if mRev < 0:
                    mRev = -self.mNuc
                else:
                    mRev = self.mNuc
            hCP = self.hCoop

        # Set up Xrm and Xrc
        xRM = np.empty_like(mSignal.independentThread.data)
        xRC = np.empty_like(mSignal.independentThread.data)
        mPosIndices = np.asarray(mSignal.independentThread.data > 0.0).nonzero()
        mNegIndices = np.asarray(mSignal.independentThread.data <= 0.0).nonzero()
        if mRev > 0.0:
            xRM[mPosIndices] = 1.0 + np.power(m[mPosIndices], 2)
            xRM[mNegIndices] = xR[mNegIndices]
            xRC[mPosIndices] = xR[mPosIndices]
            xRC[mNegIndices] = 1.0 + np.power(m[mNegIndices], 2)
        else:
            xRM[mNegIndices] = 1.0 + np.power(m[mNegIndices], 2)
            xRM[mPosIndices] = xR[mPosIndices]
            xRC[mNegIndices] = xR[mNegIndices]
            xRC[mPosIndices] = 1.0 + np.power(m[mPosIndices], 2)

        # Calculate Xexp
        xExp = xR * self.xInit
        belowNucIndices = np.asarray(np.abs(mSignal.independentThread.data) <= self.mNuc).nonzero()
        exponents = dM[belowNucIndices] / (self.xInit * (np.divide(hCP,np.float_power(xRC[belowNucIndices], 2.5)) + np.divide(self.hAnh*dM[belowNucIndices], self.mSat*np.float_power(xRM[belowNucIndices], 2.0))))
        xExp[belowNucIndices] = np.multiply(xExp[belowNucIndices], np.exp(exponents))

        # Calculate H
        # hThread = integralTrapQuadrature(1.0/xExp, mSignal.independentThread.data, hRev)
        hThread = integralTrapQuadrature(1.0/xExp, mSignal.independentThread.data, hRev)

        # Compile SignalBundle
        signalBundleArray = np.vstack((mSignal.dependentThread.data, mSignal.independentThread.data, hThread.data, xExp))
        return HysteresisSignalBundle(SignalBundle.fromSignalBundleArray(signalBundleArray, ['M','H','X']))
