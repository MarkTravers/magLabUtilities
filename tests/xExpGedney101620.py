#!python3

import numpy as np
from magLabUtilities.signalutilities.signals import SignalThread, Signal, SignalBundle
from magLabUtilities.signalutilities.hysteresis import HysteresisSignalBundle, XExpGendey101620
from magLabUtilities.uiutilities.plotting.hysteresis import MofHXofMPlotter

if __name__=='__main__':
    # xInit:np.float64, mSat:np.float64, mNuc:np.float64, hCoercive:np.float64, hAnh:np.float64, hCoop:np.float64
    xInit = 63.0
    mSat = 1.60e6
    mNuc = 1.44e6
    hCoercive = 600
    hAnh = 5800
    hCoop = 750

    amplitude = 1.51637e6

    tVirginThread = SignalThread(np.linspace(0.0, 1.0, 10000, endpoint=False))
    mVirginThread = SignalThread(np.linspace(0.0, amplitude, 10000, endpoint=False))
    mVirginSignal = Signal.fromThreadPair(mVirginThread, tVirginThread)

    tPRThread = SignalThread(np.linspace(1.0, 3.0, 20000, endpoint=False))
    mPRThread = SignalThread(np.linspace(amplitude, -amplitude, 20000, endpoint=False))
    mPRSignal = Signal.fromThreadPair(mPRThread, tPRThread)

    tNRThread = SignalThread(np.linspace(2.0, 5.0, 20000))
    mNRThread = SignalThread(np.linspace(-amplitude, amplitude, 20000))
    mNRSignal = Signal.fromThreadPair(mNRThread, tNRThread)

    xExp = XExpGendey101620(xInit, mSat, mNuc, hCoercive, hAnh, hCoop)

    virginBundle = xExp.evaluate(mVirginSignal, mRev=0.0, hRev=0.0, curveRegion='virgin')
    prBundle = xExp.evaluate(mPRSignal, mRev=amplitude, hRev=np.amax(virginBundle.signals['H'].independentThread.data), curveRegion='reversal')
    nrBundle = xExp.evaluate(mNRSignal, mRev=-amplitude, hRev=np.amin(prBundle.signals['H'].independentThread.data), curveRegion='reversal')

    plotter = MofHXofMPlotter()
    plotter.addMofHPlot(virginBundle, 'virgin')
    plotter.addMofHPlot(prBundle, 'Positive Reversal')
    plotter.addMofHPlot(nrBundle, 'Negative Reversal')
    plotter.addXofMPlot(virginBundle, 'virgin')
    plotter.addXofMPlot(prBundle, 'Positive Reversal')
    plotter.addXofMPlot(nrBundle, 'Negative Reversal')
    
    print('done.')
