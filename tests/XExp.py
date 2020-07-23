#!python3

import numpy as np
from magLabUtilities.signalutilities.signals import SignalThread, Signal, SignalBundle
from magLabUtilities.signalutilities.hysteresis import HysteresisSignalBundle, XExpOfHGedney071720
from magLabUtilities.signalutilities.canonical1d import Line
from magLabUtilities.signalutilities.calculus import integralIndexQuadrature
from magLabUtilities.uiutilities.plotting.hysteresis import MofHXofMPlotter

if __name__=='__main__':
    hAmp = 20000

    xInit = 67.0
    hCoercive = 700.0
    hNuc = 20000
    mSat = 1.66e6
    hCoop = 3130.0
    hNuc = 11974
    mNuc = 1.5221e6
    hAnh = 4300.0

    virginHThread = Line(x0=0.0, x1=hAmp, t0=0.0, t1=10000.0)
    pRevHThread = Line(x0=hAmp, x1=-hAmp, t0=10000.0, t1=30000.0)
    nRevHThread = Line(x0=-hAmp, x1=hAmp, t0=30000.0, t1=50000.0)

    # xInit:float, hCoercive:float, hNuc:float, mNuc:float, mSat:float, hCoop:float, hAnh:float, hRev:float, mRev:float, curveRegion:str
    xExpGen = XExpOfHGedney071720(xInit=xInit, hCoercive=hCoercive, hNuc=hNuc, mNuc=mNuc, mSat=mSat, hCoop=hCoop, hAnh=hAnh)
    virginX = xExpGen.evaluate(hSignal=virginHThread.evaluate(tThread=SignalThread(np.linspace(0.0, 10000.0, 10001))), hRev=0.0, mRev=0.0, curveRegion='virgin')

    xExpGen.hCoercive = hCoop
    xExpGen.hCoop = 0.0
    hRev = np.amax(virginX.signals['H'].independentThread.data)
    mRev = np.amax(virginX.signals['M'].independentThread.data)
    pRevX = xExpGen.evaluate(hSignal=pRevHThread.evaluate(tThread=SignalThread(np.linspace(10001.0, 30000.0, 20000))), hRev=hRev, mRev=mRev, curveRegion='reversal')

    hRev = np.amin(pRevX.signals['H'].independentThread.data)
    mRev = np.amin(pRevX.signals['M'].independentThread.data)
    nRevX = xExpGen.evaluate(hSignal=nRevHThread.evaluate(tThread=SignalThread(np.linspace(30001.0, 50000.0, 20000))), hRev=hRev, mRev=mRev, curveRegion='reversal')

    hysteresisBundle = HysteresisSignalBundle.fromSignalBundleSequence([virginX, pRevX, nRevX])

    # # Take the integral of the model
    # hThread = SignalThread(integralIndexQuadrature(1.0 / hysteresisBundle.signals['X'].independentThread.data, hysteresisBundle.signals['M'].independentThread.data))
    # hysteresisBundle.addSignal('H', Signal.fromThreadPair(hThread, hysteresisBundle.signals['M'].dependentThread))

    plotter = MofHXofMPlotter()
    plotter.addMofHPlot(hysteresisBundle, 'Xexp')
    plotter.addXofMPlot(hysteresisBundle, 'Xexp')
    plotter.addXRevofMPlot(hysteresisBundle, 'Xrev')

    print('done')
