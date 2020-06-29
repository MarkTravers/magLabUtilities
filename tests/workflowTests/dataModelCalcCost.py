#!python3

import numpy as np
from magLabUtilities.datafileutilities.timeDomain import importFromXlsx
from magLabUtilities.signalutilities.signals import SignalThread, Signal, SignalBundle
from magLabUtilities.signalutilities.hysteresis import XExpQA, HysteresisSignalBundle
from magLabUtilities.uiutilities.plotting.hysteresis import MofHPlotter, XofMPlotter

if __name__=='__main__':
    fp = './tests/workflowTests/datafiles/test21kLoop.xlsx'
    refBundle = HysteresisSignalBundle(importFromXlsx(fp, '21k', 2, 'C,D', dataColumnNames=['H','M']))

    mMatrix = refBundle.signals['M'].independentThread.data
    tMatrix = refBundle.signals['M'].dependentThread.data
    pMAmpIndex = np.argmax(mMatrix[0:int(mMatrix.shape[0]/2)])
    nMAmpIndex = np.argmin(mMatrix)

    virginGen = XExpQA(xInit=67.0, hCoercive=630.0, mSat=1.67e6, hCoop=1200.0, hAnh=3000.0, xcPow=4.0, mRev=0.0)
    pRevGen = XExpQA(xInit=67.0, hCoercive=630.0, mSat=1.67e6, hCoop=1200.0, hAnh=3000.0, xcPow=4.0, mRev=mMatrix[pMAmpIndex])
    nRevGen = XExpQA(xInit=67.0, hCoercive=630.0, mSat=1.67e6, hCoop=1200.0, hAnh=3000.0, xcPow=4.0, mRev=mMatrix[nMAmpIndex])

    virginM = Signal.fromThreadPair(SignalThread(mMatrix[0:pMAmpIndex]), SignalThread(tMatrix[0:pMAmpIndex]))
    virginX = virginGen.evaluate(mSignal=virginM)

    pRevM = Signal.fromThreadPair(SignalThread(mMatrix[pMAmpIndex:nMAmpIndex]), SignalThread(tMatrix[pMAmpIndex:nMAmpIndex]))
    pRevX = pRevGen.evaluate(mSignal=pRevM)
    
    nRevM = Signal.fromThreadPair(SignalThread(mMatrix[nMAmpIndex:]), SignalThread(tMatrix[nMAmpIndex:]))
    nRevX = nRevGen.evaluate(mSignal=nRevM)

    testBundle = HysteresisSignalBundle.fromSignalBundleSequence([virginX, pRevX, nRevX])

    plotter = XofMPlotter()
    # plotter.addPlot(refBundle, 'Data')
    plotter.addPlot(virginX, 'Virgin')
    plotter.addPlot(pRevX, 'Positive Reversal')
    plotter.addPlot(nRevX, 'Negative Reversal')

    input('Press Return to exit...')
    print('done')
