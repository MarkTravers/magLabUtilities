#!python3

from typing import List
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
    def __init__(self, dataFP:str, loopNameList:List[str], tuneHistoryFP:str):
        self.loopNameList = loopNameList
        self.tuneHistoryFP = tuneHistoryFP

        self.loopDict = {}
        interpolator = Legendre(interpRadius=100, legendreOrder=3)
        tThread = SignalThread(np.linspace(0.0, 5.0, 500))
        self.plotter = MofHXofMPlotter()

        for sheetName in loopNameList:
            self.loopDict[sheetName] = {}
            # Import curve
            bundle = HysteresisSignalBundle(importFromXlsx(dataFP, sheetName, 1, 'A,B', dataColumnNames=['H','M']))
            # Re-parameterize data by arc length
            bundleArray = np.vstack([bundle.signals['H'].dependentThread.data, bundle.signals['H'].independentThread.data, bundle.signals['M'].independentThread.data])
            bundleArray = SignalBundle.arcLengthND(bundleArray, totalArcLength=5.0)
            bundle = HysteresisSignalBundle.fromSignalBundleArray(bundleArray, ['H', 'M'])
            # Re-sample data fore more even arc length
            bundle.signals['M'] = interpolator.interpolate(bundle.signals['M'], tThread)
            bundle.signals['H'] = interpolator.interpolate(bundle.signals['H'], tThread)
            # Find indices of reversals
            pMAmpIndex = np.argmax(bundle.signals['M'].independentThread.data[0:int(bundle.signals['M'].independentThread.data.size / 2)])
            nMAmpIndex = np.argmin(bundle.signals['M'].independentThread.data)
            # Calculate susceptibility of loop
            xThread = SignalThread(finiteDiffDerivative( \
                fNum=bundle.signals['M'].independentThread.data, \
                fDenom=bundle.signals['H'].independentThread.data, \
                windowRadius=1, \
                discontinuousPoints=[pMAmpIndex, nMAmpIndex], \
                differenceMode='centralDifference'))
            bundle.addSignal('X', Signal.fromThreadPair(xThread, bundle.signals['M'].dependentThread))
            # Plot loop
            self.plotter.addMofHPlot(bundle, sheetName)
            self.plotter.addXofMPlot(bundle, sheetName)

            # Add loop data to dict
            self.loopDict[sheetName]['bundle'] = bundle
            self.loopDict[sheetName]['pMAmpIndex'] = pMAmpIndex
            self.loopDict[sheetName]['nMAmpIndex'] = nMAmpIndex

    def runCostFunction(self, gridNode:GridNode) -> GridNode:
        # xInit:float, hCoercive:float, hNuc:float, mNuc:float, mSat:float, hCoop:float, hAnh:float
        xInit = gridNode.coordList[0]
        hCoercive = gridNode.coordList[1]
        hNuc = gridNode.coordList[2]
        mNuc = gridNode.coordList[3]
        mSat = gridNode.coordList[4]
        hAnh = gridNode.coordList[5]
        hCoop = {}
        hCoop['22.7k'] = gridNode.coordList[6]
        hCoop['9.4k'] = gridNode.coordList[7]
        hCoop['1.1k'] = gridNode.coordList[8]
        hCoop['0.6k'] = gridNode.coordList[9]
        hCoop['0.5k'] = gridNode.coordList[10]

        losses = []
        testLoopDict = {}

        for loopName in self.loopNameList:
            # Configure input H-threads
            virginH = Signal.fromThreadPair(SignalThread(self.loopDict[loopName]['bundle'].signals['H'].independentThread.data[0:self.loopDict[loopName]['pMAmpIndex']]), SignalThread(self.loopDict[loopName]['bundle'].signals['H'].dependentThread.data[0:self.loopDict[loopName]['pMAmpIndex']]))
            pRevH = Signal.fromThreadPair(SignalThread(self.loopDict[loopName]['bundle'].signals['H'].independentThread.data[self.loopDict[loopName]['pMAmpIndex']:self.loopDict[loopName]['nMAmpIndex']]), SignalThread(self.loopDict[loopName]['bundle'].signals['H'].dependentThread.data[self.loopDict[loopName]['pMAmpIndex']:self.loopDict[loopName]['nMAmpIndex']]))
            nRevH = Signal.fromThreadPair(SignalThread(self.loopDict[loopName]['bundle'].signals['H'].independentThread.data[self.loopDict[loopName]['nMAmpIndex']:]), SignalThread(self.loopDict[loopName]['bundle'].signals['H'].dependentThread.data[self.loopDict[loopName]['nMAmpIndex']:]))
            # Configure Xexp generator
            xExpGen = XExpOfHGedney071720(xInit=xInit, hCoercive=hCoercive, hNuc=hNuc, mNuc=mNuc, mSat=mSat, hCoop=hCoop[loopName], hAnh=hAnh)
            # Evaluate Xexp along loop
            hRev = np.amin(self.loopDict[loopName]['bundle'].signals['H'].independentThread.data)
            mRev = np.amin(self.loopDict[loopName]['bundle'].signals['M'].independentThread.data)
            virginX = xExpGen.evaluate(hSignal=virginH, hRev=0.0, mRev=0.0, curveRegion='virgin')
            hRev = np.amax(virginX.signals['H'].independentThread.data)
            mRev = np.amax(virginX.signals['M'].independentThread.data)
            pRevX = xExpGen.evaluate(hSignal=pRevH, hRev=hRev, mRev=mRev, curveRegion='reversal')
            hRev = np.amin(pRevX.signals['H'].independentThread.data)
            mRev = np.amin(pRevX.signals['M'].independentThread.data)
            nRevX = xExpGen.evaluate(hSignal=nRevH, hRev=hRev, mRev=mRev, curveRegion='reversal')
            # Compile curve regions into one signalBundle
            testBundle = HysteresisSignalBundle.fromSignalBundleSequence([virginX, pRevX, nRevX])
            # Calculate loss
            refMatrix = self.loopDict[loopName]['bundle'].sample(tThread=self.loopDict[loopName]['bundle'].signals['H'].dependentThread, signalInterpList=[('M',nearestPoint),('H',nearestPoint)])
            testMatrix = testBundle.sample(tThread=self.loopDict[loopName]['bundle'].signals['H'].dependentThread, signalInterpList=[('M',nearestPoint),('H',nearestPoint)])
            tWeightMatrix = np.vstack([self.loopDict[loopName]['bundle'].signals['H'].dependentThread.data, np.ones(500)])
            losses.append(rmsNdNorm(refMatrix, testMatrix, tWeightMatrix))
            testLoopDict[loopName] = testBundle

        gridNode.loss = np.sum(np.asarray(losses, dtype=np.float64))
        gridNode.data = testLoopDict
        return gridNode

    def gradientStep(self, newCenterGridNode):
        for loopName in self.loopNameList:
            self.plotter.addMofHPlot(newCenterGridNode.data[loopName], 'Model')
            self.plotter.addXofMPlot(newCenterGridNode.data[loopName], 'Model')
            self.plotter.addXRevofMPlot(newCenterGridNode.data[loopName], 'Xrev')

        with open(tuneHistoryFP, 'a') as tuneHistoryFile:
            tuneHistoryFile.write(str(datetime.fromtimestamp(datetime.timestamp(datetime.now()))) + '\n')
            tuneHistoryFile.write(str(newCenterGridNode.coordList) + '\n')
            tuneHistoryFile.write('Error: %s\n' % str(newCenterGridNode.loss))
            # tuneHistoryFile.write(json.dumps(newCenterGridNode.data) + '\n')

        print(newCenterGridNode.loss)
        print('Switching to node: %s' % str(newCenterGridNode.coordList))

if __name__ == '__main__':
    # xInit = gridNode.coordList[0]
    # hCoercive = gridNode.coordList[1]
    # hNuc = gridNode.coordList[2]
    # mNuc = gridNode.coordList[3]
    # mSat = gridNode.coordList[4]
    # hAnh = gridNode.coordList[5]
    # hCoop = {}
    # hCoop['22.7k'] = gridNode.coordList[6]
    # hCoop['9.4k'] = gridNode.coordList[7]
    # hCoop['1.1k'] = gridNode.coordList[8]
    # hCoop['0.6k'] = gridNode.coordList[9]
    # hCoop['0.5k'] = gridNode.coordList[10]

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
                            'testGridLocalIndices':[0]
                            # 'testGridLocalIndices':[-1,0,1]
                        },
                        {   'name':'mNuc',
                            'initialValue':1.5221e6,
                            'stepSize':0.03e6,
                            'testGridLocalIndices':[0]
                            # 'testGridLocalIndices':[-1,0,1]
                        },
                        {   'name':'mSat',
                            'initialValue':1.66e6,
                            'stepSize':0.01e6,
                            'testGridLocalIndices':[0]
                            # 'testGridLocalIndices':[-1,0,1]
                        },
                        {   'name':'hAnh',
                            'initialValue':4300.0,
                            'stepSize':10.0,
                            'testGridLocalIndices':[0]
                            # 'testGridLocalIndices':[-1,0,1]
                        },
                        {   'name':'hCoop22.7k',
                            'initialValue':660.0,
                            'stepSize':5.0,
                            'testGridLocalIndices':[0]
                            # 'testGridLocalIndices':[-1,0,1]
                        },
                        {   'name':'hCoop9.4k',
                            'initialValue':565.0,
                            'stepSize':5.0,
                            'testGridLocalIndices':[0]
                            # 'testGridLocalIndices':[-1,0,1]
                        },
                        {   'name':'hCoop1.1k',
                            'initialValue':435.0,
                            'stepSize':5.0,
                            'testGridLocalIndices':[0]
                            # 'testGridLocalIndices':[-1,0,1]
                        },
                        {   'name':'hCoop0.6k',
                            'initialValue':675.0,
                            'stepSize':5.0,
                            'testGridLocalIndices':[0]
                            # 'testGridLocalIndices':[-1,0,1]
                        },
                        {   'name':'hCoop0.5k',
                            'initialValue':695.0,
                            'stepSize':5.0,
                            'testGridLocalIndices':[0]
                            # 'testGridLocalIndices':[-1,0,1]
                        }
                    ]

    fp = './tests/workflowTests/datafiles/CarlData.xlsx'
    loopNameList = ['22.7k', '9.4k', '1.1k', '0.6k', '0.5k']
    tuneHistoryFP = './tests/workflowTests/datafiles/tuneHistory01.txt'

    costEvaluator = CostEvaluator(fp, loopNameList, tuneHistoryFP)
    tuner = GradientDescent(parameterList, costEvaluator.runCostFunction, costEvaluator.gradientStep)
    tuner.tune(numIterations=1, maxThreads=81)

    print('done')
