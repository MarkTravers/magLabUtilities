#!python

import numpy as np
from magLabUtilities.datafileutilities.timeDomain import importFromXlsx
from magLabUtilities.signalutilities.signals import SignalThread, Signal, SignalBundle
from magLabUtilities.signalutilities.hysteresis import XExpQA, HysteresisSignalBundle
from magLabUtilities.uiutilities.plotting.hysteresis import MofHPlotter, XofMPlotter, MofHXofMPlotter
from magLabUtilities.signalutilities.calculus import finiteDiffDerivative, integralIndexQuadrature

if __name__=='__main__':
    # Import data
    fp = './tests/workflowTests/datafiles/test21kLoop.xlsx'
    refBundle = HysteresisSignalBundle(importFromXlsx(fp, '21k', 2, 'C,D', dataColumnNames=['H','M']))

    # Generate a Xexp which matches the datafile
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

    # Take the derivative of the data
    xThread = SignalThread(finiteDiffDerivative( \
        fNum=refBundle.signals['M'].independentThread.data, \
        fDenom=refBundle.signals['H'].independentThread.data, \
        windowRadius=100, \
        discontinuousPoints=[pMAmpIndex, nMAmpIndex], \
        differenceMode='centralDifference'))
    refBundle.addSignal('X', Signal.fromThreadPair(xThread, refBundle.signals['M'].dependentThread))

    # Take the integral of the model
    hThread = SignalThread(integralIndexQuadrature(1.0 / testBundle.signals['X'].independentThread.data, testBundle.signals['M'].independentThread.data))
    testBundle.addSignal('H', Signal.fromThreadPair(hThread, testBundle.signals['M'].dependentThread))

    plotter = MofHXofMPlotter()
    plotter.addMofHPlot(refBundle, 'Data')
    plotter.addMofHPlot(testBundle, 'Model')
    plotter.addXofMPlot(refBundle, 'Data')
    plotter.addXofMPlot(testBundle, 'Model')
    # plotter.addXofMPlot(virginX, 'Virgin')
    # plotter.addXofMPlot(pRevX, 'Positive Reversal')
    # plotter.addXofMPlot(nRevX, 'Negative Reversal')

    # input('Press Return to exit...')
    print('done')
