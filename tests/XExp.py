#!python3

import numpy as np
from magLabUtilities.signalutilities.signals import SignalThread, Signal, SignalBundle
from magLabUtilities.signalutilities.hysteresis import HysteresisSignalBundle, XExpQA
from magLabUtilities.signalutilities.canonical1d import Line
from magLabUtilities.uiutilities.plotting.hysteresis import XofMPlotter

if __name__=='__main__':
    mAmp = 1.0e6

    virginGen = XExpQA(xInit=67.0, hCoercive=630.0, mSat=1.67e6, hCoop=1200.0, hAnh=3000.0, xcPow=4.0, mRev=0.0)
    pRevGen = XExpQA(xInit=67.0, hCoercive=630.0, mSat=1.67e6, hCoop=1200.0, hAnh=3000.0, xcPow=4.0, mRev=mAmp)
    nRevGen = XExpQA(xInit=67.0, hCoercive=630.0, mSat=1.67e6, hCoop=1200.0, hAnh=3000.0, xcPow=4.0, mRev=-mAmp)

    virginMThread = Line(x0=0.0, x1=mAmp, t0=0.0, t1=50.0)
    pRevMThread = Line(x0=mAmp, x1=-mAmp, t0=0.0, t1=100.0)
    nRevMThread = Line(x0=-mAmp, x1=mAmp, t0=0.0, t1=100.0)

    virginX = virginGen.evaluate(mSignal=virginMThread.evaluate(tThread=SignalThread(np.linspace(0.0, 50.0, 50))))
    pRevX = pRevGen.evaluate(mSignal=pRevMThread.evaluate(tThread=SignalThread(np.linspace(0.0, 100.0, 100))))
    nRevX = nRevGen.evaluate(mSignal=nRevMThread.evaluate(tThread=SignalThread(np.linspace(0.0, 100.0, 100))))

    loop = HysteresisSignalBundle.fromSignalBundleSequence([virginX, pRevX, nRevX])

    plotter = XofMPlotter()
    plotter.addPlot(loop, 'Model')

    input('Press any key to exit...')
    print('done')
