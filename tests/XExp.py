#!python3

import numpy as np
from magLabUtilities.signalutilities.signals import SignalThread, Signal, SignalBundle
from magLabUtilities.signalutilities.hysteresis import HysteresisSignalBundle, XExpGedney060820
from magLabUtilities.signalutilities.canonical1d import Line
from magLabUtilities.uiutilities.plotting.hysteresis import XofMPlotter

if __name__=='__main__':
    mAmp = 1.0e6
    virginMTolerance = 1e4

    xInit = 73.0
    hCoercive = 610.0
    mSat = 1.66e6
    hCoop = 814.0
    hAnh = 4310.0
    xcPow = 2.0

    virginGen = XExpGedney060820(xInit, hCoercive, mSat, hCoop, hAnh, xcPow, mRev=0.0, virginMTolerance=virginMTolerance)
    pRevGen = XExpGedney060820(xInit, hCoercive, mSat, hCoop, hAnh, xcPow, mRev=mAmp, virginMTolerance=virginMTolerance)
    nRevGen = XExpGedney060820(xInit, hCoercive, mSat, hCoop, hAnh, xcPow, mRev=-mAmp, virginMTolerance=virginMTolerance)

    virginMThread = Line(x0=0.0, x1=mAmp, t0=0.0, t1=100.0)
    pRevMThread = Line(x0=mAmp, x1=-mAmp, t0=0.0, t1=200.0)
    nRevMThread = Line(x0=-mAmp, x1=mAmp, t0=0.0, t1=200.0)

    virginX = virginGen.evaluate(mSignal=virginMThread.evaluate(tThread=SignalThread(np.linspace(0.0, 100.0, 101))))
    pRevX = pRevGen.evaluate(mSignal=pRevMThread.evaluate(tThread=SignalThread(np.linspace(0.0, 200.0, 201))))
    nRevX = nRevGen.evaluate(mSignal=nRevMThread.evaluate(tThread=SignalThread(np.linspace(0.0, 200.0, 201))))

    # loop = HysteresisSignalBundle.fromSignalBundleSequence([virginX, pRevX, nRevX])

    plotter = XofMPlotter()
    plotter.addPlot(virginX, 'Virgin')
    plotter.addPlot(pRevX, 'Positive Reversal')
    plotter.addPlot(nRevX, 'Negative Reversal')

    print('done')
