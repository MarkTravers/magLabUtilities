#!python3

import numpy as np
from magLabUtilities.signalutilities.signals import SignalThread, Signal, SignalBundle
from magLabUtilities.signalutilities.hysteresis import HysteresisSignalBundle, XExpOfHGedney071720
from magLabUtilities.signalutilities.canonical1d import Line
from magLabUtilities.signalutilities.calculus import integralIndexQuadrature
from magLabUtilities.uiutilities.plotting.hysteresis import MofHXofMPlotter

if __name__=='__main__':
    hAmp = 500

    xInit = 67.0
    hCoercive = 700.0
    hNuc = 11974
    mSat = 1.66e6
    hCoop = 695.0
    hNuc = 11974
    mNuc = 1.5221e6
    hAnh = 4300.0

    virginHThread = Line(x0=0.0, x1=hAmp, t0=0.0, t1=250.0)
    pRevHThread = Line(x0=hAmp, x1=-hAmp, t0=250.0, t1=750.0)
    nRevHThread = Line(x0=-hAmp, x1=hAmp, t0=750.0, t1=1250.0)

    # xInit:float, hCoercive:float, hNuc:float, mNuc:float, mSat:float, hCoop:float, hAnh:float, hRev:float, mRev:float, curveRegion:str
    xExpGen = XExpOfHGedney071720(xInit=xInit, hCoercive=hCoercive, hNuc=hNuc, mNuc=mNuc, mSat=mSat, hCoop=hCoop, hAnh=hAnh)
    virginX = xExpGen.evaluate(hSignal=virginHThread.evaluate(tThread=SignalThread(np.linspace(0.0, 250.0, 251))), hRev=0.0, mRev=0.0, curveRegion='virgin')

    hRev = np.amax(virginX.signals['H'].independentThread.data)
    mRev = np.amax(virginX.signals['M'].independentThread.data)
    pRevX = xExpGen.evaluate(hSignal=pRevHThread.evaluate(tThread=SignalThread(np.linspace(251.0, 750.0, 500))), hRev=hRev, mRev=mRev, curveRegion='reversal')

    hRev = np.amin(pRevX.signals['H'].independentThread.data)
    mRev = np.amin(pRevX.signals['M'].independentThread.data)
    nRevX = xExpGen.evaluate(hSignal=nRevHThread.evaluate(tThread=SignalThread(np.linspace(751.0, 1250.0, 500))), hRev=hRev, mRev=mRev, curveRegion='reversal')

    hysteresisBundle = HysteresisSignalBundle.fromSignalBundleSequence([virginX, pRevX, nRevX])

    # # Take the integral of the model
    # hThread = SignalThread(integralIndexQuadrature(1.0 / hysteresisBundle.signals['X'].independentThread.data, hysteresisBundle.signals['M'].independentThread.data))
    # hysteresisBundle.addSignal('H', Signal.fromThreadPair(hThread, hysteresisBundle.signals['M'].dependentThread))

    plotter = MofHXofMPlotter()
    plotter.addMofHPlot(hysteresisBundle, 'Xexp')
    plotter.addXofMPlot(hysteresisBundle, 'Xexp')
    plotter.addXRevofMPlot(hysteresisBundle, 'Xrev')

    print('done')
