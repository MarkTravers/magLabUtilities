#!python3

import numpy as np
from magLabUtilities.signalutilities.signals import SignalThread, Signal, SignalBundle
from magLabUtilities.datafileutilities.timeDomain import importFromXlsx
from magLabUtilities.signalutilities.interpolation import Legendre, nearestPoint
from magLabUtilities.signalutilities.hysteresis import HysteresisSignalBundle, XExpOfHGedney071720
from magLabUtilities.optimizerutilities.costFunctions import rmsNdNorm
from magLabUtilities.optimizerutilities.parameterSpaces import GridNode
from magLabUtilities.optimizerutilities.gradientDescent import GradientDescent
from magLabUtilities.signalutilities.calculus import finiteDiffDerivative, integralIndexQuadrature
from magLabUtilities.uiutilities.plotting.hysteresis import MofHXofMPlotter
from datetime import datetime
import json

class CostEvaluator:
    def __init__(self, dataFP, tuneHistoryFP):
        # Import data
        self.fp = dataFP
        self.refBundle = HysteresisSignalBundle(importFromXlsx(self.fp, '9.4k', 1, 'A,B', dataColumnNames=['H','M']))
        # Re-parameterize data by arc length
        refBundleArray = np.vstack([self.refBundle.signals['H'].dependentThread.data, self.refBundle.signals['H'].independentThread.data, self.refBundle.signals['M'].independentThread.data])
        refBundleArray = SignalBundle.arcLengthND(refBundleArray, totalArcLength=5.0)
        self.refBundle = HysteresisSignalBundle.fromSignalBundleArray(refBundleArray, ['H', 'M'])
        # Re-sample data for more even arc length
        interpolator = Legendre(interpRadius=100, legendreOrder=3)
        tThread = SignalThread(np.linspace(0.0, 5.0, 1000))
        self.refBundle.signals['M'] = interpolator.interpolate(self.refBundle.signals['M'], tThread)
        self.refBundle.signals['H'] = interpolator.interpolate(self.refBundle.signals['H'], tThread)
        # Find indices of reversals
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
        # xInit:float, hCoercive:float, hNuc:float, mNuc:float, mSat:float, hCoop:float, hAnh:float
        xInit = gridNode.coordList[0]
        hCoercive = gridNode.coordList[1]
        hNuc = gridNode.coordList[2]
        mNuc = gridNode.coordList[3]
        mSat = gridNode.coordList[4]
        hCoop = gridNode.coordList[5]
        hAnh = gridNode.coordList[6]

        # Configure input H-threads
        virginH = Signal.fromThreadPair(SignalThread(self.refBundle.signals['H'].independentThread.data[0:self.pMAmpIndex]), SignalThread(self.refBundle.signals['H'].dependentThread.data[0:self.pMAmpIndex]))
        pRevH = Signal.fromThreadPair(SignalThread(self.refBundle.signals['H'].independentThread.data[self.pMAmpIndex:self.nMAmpIndex]), SignalThread(self.refBundle.signals['H'].dependentThread.data[self.pMAmpIndex:self.nMAmpIndex]))
        nRevH = Signal.fromThreadPair(SignalThread(self.refBundle.signals['H'].independentThread.data[self.nMAmpIndex:]), SignalThread(self.refBundle.signals['H'].dependentThread.data[self.nMAmpIndex:]))
        # Configure Xexp generator
        xExpGen = XExpOfHGedney071720(xInit=xInit, hCoercive=hCoercive, hNuc=hNuc, mNuc=mNuc, mSat=mSat, hCoop=hCoop, hAnh=hAnh)
        # Evaluate Xexp along loop
        hRev = np.amin(self.refBundle.signals['H'].independentThread.data)
        mRev = np.amin(self.refBundle.signals['M'].independentThread.data)
        virginX = xExpGen.evaluate(hSignal=virginH, hRev=0.0, mRev=0.0, curveRegion='virgin')
        hRev = np.amax(virginX.signals['H'].independentThread.data)
        mRev = np.amax(virginX.signals['M'].independentThread.data)
        pRevX = xExpGen.evaluate(hSignal=pRevH, hRev=hRev, mRev=mRev, curveRegion='reversal')
        hRev = np.amin(pRevX.signals['H'].independentThread.data)
        mRev = np.amin(pRevX.signals['M'].independentThread.data)
        nRevX = xExpGen.evaluate(hSignal=nRevH, hRev=hRev, mRev=mRev, curveRegion='reversal')
        # Compile curve regions into one signalBundle
        testBundle = HysteresisSignalBundle.fromSignalBundleSequence([virginX, pRevX, nRevX])

        refMatrix = self.refBundle.sample(tThread=self.refBundle.signals['H'].dependentThread, signalInterpList=[('M',nearestPoint),('H',nearestPoint)])
        testMatrix = testBundle.sample(tThread=self.refBundle.signals['H'].dependentThread, signalInterpList=[('M',nearestPoint),('H',nearestPoint)])
        tWeightMatrix = np.vstack([self.refBundle.signals['H'].dependentThread.data, np.hstack([np.ones(200), np.ones(800)])])
        gridNode.loss = rmsNdNorm(refMatrix, testMatrix, tWeightMatrix)
        gridNode.data = testBundle
        return gridNode

    def gradientStep(self, newCenterGridNode):
        self.plotter.addMofHPlot(newCenterGridNode.data, 'Model')
        self.plotter.addXofMPlot(newCenterGridNode.data, 'Model')
        self.plotter.addXRevofMPlot(newCenterGridNode.data, 'Xrev')

        # with open(tuneHistoryFP, 'a') as tuneHistoryFile:
        #     tuneHistoryFile.write(str(datetime.fromtimestamp(datetime.timestamp(datetime.now()))) + '\n')
        #     tuneHistoryFile.write(str(newCenterGridNode.coordList) + '\n')
        #     tuneHistoryFile.write('Error: %s\n' % str(newCenterGridNode.loss))
        #     tuneHistoryFile.write(json.dumps(newCenterGridNode.data) + '\n')

        print(newCenterGridNode.loss)
        print('Switching to node: %s' % str(newCenterGridNode.coordList))

if __name__ == '__main__':
    # xInit = gridNode.coordList[0]
    # hCoercive = gridNode.coordList[1]
    # hNuc = gridNode.coordList[2]
    # mNuc = gridNode.coordList[3]
    # mSat = gridNode.coordList[4]
    # hCoop = gridNode.coordList[5]
    # hAnh = gridNode.coordList[6]
    parameterList = [
                        {   'name':'xInit',
                            'initialValue':69.0,
                            'stepSize':3,
                            'testGridLocalIndices':[0]
                            # 'testGridLocalIndices':[-1,0,1]
                        },
                        {   'name':'hCoercive',
                            'initialValue':680.0,
                            'stepSize':25.0,
                            'testGridLocalIndices':[0]
                            # 'testGridLocalIndices':[-1,0,1]
                        },
                        {   'name':'hNuc',
                            'initialValue':11974.0,
                            'stepSize':200,
                            # 'testGridLocalIndices':[0]
                            'testGridLocalIndices':[-1,0,1]
                        },
                        {   'name':'mNuc',
                            'initialValue':1.5221e6,
                            'stepSize':0.03e6,
                            # 'testGridLocalIndices':[0]
                            'testGridLocalIndices':[-1,0,1]
                        },
                        {   'name':'mSat',
                            'initialValue':1.66e6,
                            'stepSize':0.01e6,
                            'testGridLocalIndices':[0]
                            # 'testGridLocalIndices':[-1,0,1]
                        },
                        {
                            'name':'hCoop',
                            'initialValue':660.0,
                            'stepSize':50.0,
                            # 'testGridLocalIndices':[0]
                            'testGridLocalIndices':[-1,0,1]
                        },
                        {
                            'name':'hAnh',
                            'initialValue':4300.0,
                            'stepSize':50.0,
                            # 'testGridLocalIndices':[0]
                            'testGridLocalIndices':[-1,0,1]
                        }
                    ]

    fp = './tests/workflowTests/datafiles/CarlData.xlsx'
    tuneHistoryFP = './tests/workflowTests/datafiles/tuneHistory01.txt'

    costEvaluator = CostEvaluator(fp, tuneHistoryFP)
    tuner = GradientDescent(parameterList, costEvaluator.runCostFunction, costEvaluator.gradientStep)
    tuner.tune(numIterations=np.infty, maxThreads=8)

    print('done')
