#!python3

import numpy as np
from magLabUtilities.signalutilities.signals import SignalThread, Signal, SignalBundle
from magLabUtilities.datafileutilities.timeDomain import importFromXlsx
from magLabUtilities.signalutilities.interpolation import legendre
from magLabUtilities.signalutilities.hysteresis import HysteresisSignalBundle, XExpGedney060820
from magLabUtilities.optimizerutilities.costFunctions import rmsNdNorm
from magLabUtilities.optimizerutilities.testCases import GridNode
from magLabUtilities.optimizerutilities.gradientDescent import GradientDescent
from magLabUtilities.signalutilities.calculus import finiteDiffDerivative, integralIndexQuadrature
from magLabUtilities.uiutilities.plotting.hysteresis import MofHXofMPlotter
from datetime import datetime
import json

class CostEvaluator:
    def __init__(self, dataFP, tuneHistoryFP):
        self.fp = dataFP
        self.refBundle = HysteresisSignalBundle(importFromXlsx(self.fp, '21k', 2, 'C,D', dataColumnNames=['H','M']))
        self.refBundle.signals['M'].independentThread.data = legendre(self.refBundle.signals['M'].independentThread.data, integrationWindowSize=100, stepSize=25, legendreOrder=2)
        self.refBundle.signals['M'].dependentThread.data = legendre(self.refBundle.signals['M'].dependentThread.data, integrationWindowSize=100, stepSize=25, legendreOrder=2)
        self.refBundle.signals['H'].independentThread.data = legendre(self.refBundle.signals['H'].independentThread.data, integrationWindowSize=100, stepSize=25, legendreOrder=2)
        self.refBundle.signals['H'].dependentThread.data = legendre(self.refBundle.signals['H'].dependentThread.data, integrationWindowSize=100, stepSize=25, legendreOrder=2)

        self.pMAmpIndex = np.argmax(self.refBundle.signals['M'].independentThread.data[0:int(self.refBundle.signals['M'].independentThread.data.shape[0]/2)])
        self.nMAmpIndex = np.argmin(self.refBundle.signals['M'].independentThread.data)

        # Take the derivative of the data
        xThread = SignalThread(finiteDiffDerivative( \
            fNum=self.refBundle.signals['M'].independentThread.data, \
            fDenom=self.refBundle.signals['H'].independentThread.data, \
            windowRadius=1, \
            discontinuousPoints=[self.pMAmpIndex, self.nMAmpIndex], \
            differenceMode='centralDifference'))
        self.refBundle.addSignal('X', Signal.fromThreadPair(xThread, self.refBundle.signals['M'].dependentThread))

        self.tuneHistoryFP = tuneHistoryFP

        self.plotter = MofHXofMPlotter()
        self.plotter.addMofHPlot(self.refBundle, 'Data')
        self.plotter.addXofMPlot(self.refBundle, 'Data')

    def runCostFunction(self, gridNode:GridNode) -> GridNode:
        hCoercive = gridNode.coordList[0]
        xInit = gridNode.coordList[1]
        mSat = gridNode.coordList[2]
        hCoop = gridNode.coordList[3]
        hAnh = gridNode.coordList[4]
        xcPow = gridNode.coordList[5]

        virginGen = XExpGedney060820(xInit=xInit, hCoercive=hCoercive, mSat=mSat, hCoop=hCoop, hAnh=hAnh, xcPow=xcPow, mRev=-2500.0, virginMTolerance=10000)
        pRevGen = XExpGedney060820(xInit=xInit, hCoercive=hCoercive, mSat=mSat, hCoop=hCoop, hAnh=hAnh, xcPow=xcPow, mRev=self.refBundle.signals['M'].independentThread.data[self.pMAmpIndex], virginMTolerance=10000)
        nRevGen = XExpGedney060820(xInit=xInit, hCoercive=hCoercive, mSat=mSat, hCoop=hCoop, hAnh=hAnh, xcPow=xcPow, mRev=self.refBundle.signals['M'].independentThread.data[self.nMAmpIndex], virginMTolerance=10000)

        virginM = Signal.fromThreadPair(SignalThread(self.refBundle.signals['M'].independentThread.data[0:self.pMAmpIndex]), SignalThread(self.refBundle.signals['M'].dependentThread.data[0:self.pMAmpIndex]))
        virginX = virginGen.evaluate(mSignal=virginM)

        pRevM = Signal.fromThreadPair(SignalThread(self.refBundle.signals['M'].independentThread.data[self.pMAmpIndex:self.nMAmpIndex]), SignalThread(self.refBundle.signals['M'].dependentThread.data[self.pMAmpIndex:self.nMAmpIndex]))
        pRevX = pRevGen.evaluate(mSignal=pRevM)
        
        nRevM = Signal.fromThreadPair(SignalThread(self.refBundle.signals['M'].independentThread.data[self.nMAmpIndex:]), SignalThread(self.refBundle.signals['M'].dependentThread.data[self.nMAmpIndex:]))
        nRevX = nRevGen.evaluate(mSignal=nRevM)

        testBundle = HysteresisSignalBundle.fromSignalBundleSequence([virginX, pRevX, nRevX])

        # Take the integral of the model
        hThread = SignalThread(integralIndexQuadrature(1.0 / testBundle.signals['X'].independentThread.data, testBundle.signals['M'].independentThread.data))
        testBundle.addSignal('H', Signal.fromThreadPair(hThread, testBundle.signals['M'].dependentThread))

        refMatrix = self.refBundle.sample(tThread=self.refBundle.signals['M'].dependentThread, signalInterpList=[('M','nearestPoint'),('H','nearestPoint')])
        testMatrix = testBundle.sample(tThread=self.refBundle.signals['M'].dependentThread, signalInterpList=[('M','nearestPoint'),('H','nearestPoint')])
        tWeightMatrix = np.vstack([self.refBundle.signals['M'].dependentThread.data, np.hstack([np.zeros(5), np.ones((195)), np.ones((800))])])
        gridNode.loss = rmsNdNorm(refMatrix, testMatrix, tWeightMatrix)
        gridNode.data = testBundle
        return gridNode

    def gradientStep(self, newCenterGridNode):
        self.plotter.addMofHPlot(newCenterGridNode.data, 'Model')
        self.plotter.addXofMPlot(newCenterGridNode.data, 'Model')

        with open(tuneHistoryFP, 'a') as tuneHistoryFile:
            tuneHistoryFile.write(str(datetime.fromtimestamp(datetime.timestamp(datetime.now()))) + '\n')
            tuneHistoryFile.write(str(newCenterGridNode.coordList) + '\n')
            tuneHistoryFile.write('Error: %s\n' % str(newCenterGridNode.loss))
            # tuneHistoryFile.write(json.dumps(newCenterGridNode.data) + '\n')

        print(newCenterGridNode.loss)
        print('Switching to node: %s' % str(newCenterGridNode.coordList))

if __name__ == '__main__':
    parameterDefs = {
                        'hCoercive': {'initialValue':605.0,    'limits':[0.0,10000.0]},
                        'xInit':     {'initialValue':61.25,    'limits':[60.0,90.0]},
                        'mSat':      {'initialValue':1.65e6,   'limits':[1.0e6,2.0e6]},
                        'hCoop':     {'initialValue':1190.0,   'limits':[100.0,10000.0]},
                        'hAnh':      {'initialValue':5200.0,   'limits':[100.0,10000.0]},
                        'xcPow':     {'initialValue':2.0,      'limits':[0.0,10.0]}
                    }

    gradientDescentConfig = {
                                'hCoercive': {'localNeighborSteps':[-15.0, 0.0, 15.0]},
                                'xInit':     {'localNeighborSteps':[-0.25, 0.0, 0.25]},
                                'mSat':      {'localNeighborSteps':[0.001e6, 0.0, 0.001e6]},
                                'hCoop':     {'localNeighborSteps':[-15.0, 0.0, 15.0]},
                                'hAnh':      {'localNeighborSteps':[-15.0, 0.0, 15.0]},
                                'xcPow':     {'localNeighborSteps':[-0.1, 0.0, 0.1]}
                            }

    fp = './tests/workflowTests/datafiles/testLoops.xlsx'
    tuneHistoryFP = './tests/workflowTests/datafiles/tuneHistory00.txt'

    costEvaluator = CostEvaluator(fp, tuneHistoryFP)
    tuner = GradientDescent(parameterList, costEvaluator.runCostFunction, costEvaluator.gradientStep)
    tuner.tune(numIterations=np.infty, maxThreads=8)

    print('done')
