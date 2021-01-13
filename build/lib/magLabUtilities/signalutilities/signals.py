#!python3
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Any, Union, Callable
from magLabUtilities.exceptions.exceptions import SignalTypeError, SignalValueError

class SignalThread:
    def __init__(self, data:Union[np.ndarray, list]):
        # Convert various input datatypes to a Numpy array
        if isinstance(data, np.ndarray):
            self.data = data

        elif isinstance(data, list):
            self.data = np.asarray(data)

        else:
            raise SignalTypeError('Cannot create signal from supplied data type.')

        # Ensure that the Numpy array is 1-dimensional
        if len(self.data.shape) > 1:
            raise SignalTypeError('Cannot create signal from multi-dimensional array.')

    @classmethod
    def fromThreadSequence(cls, threadList):
        return cls(np.hstack([thread.data for thread in threadList]))

    @property
    def isIncreasing(self):
        return np.all(np.diff(self.data) > 0.0)

    @property
    def isMonotonicallyIncreasing(self):
        return np.all(np.diff(self.data) >= 0.0)

    @property
    def isDecreasing(self):
        return np.all(np.diff(self.data) < 0.0)

    @property
    def isMonotonicallyDecreasing(self):
        return np.all(np.diff(self.data) <= 0.0)

    @property
    def length(self):
        return self.data.size    

class Signal:
    def __init__(self, signalConstructorType:str, constructorTuple:Tuple):
        if signalConstructorType == 'like':
            # constructorTuple = [SignalThread, SignalThread]
            self.independentThread = constructorTuple[0]
            self.dependentThread = constructorTuple[1]
            self.signalType = 'discrete'

        elif signalConstructorType == 'fromThreadPair':
            # constructorTuple = [SignalThread, SignalThread]
            self.independentThread = constructorTuple[0]
            self.dependentThread = constructorTuple[1]
            self.signalType = 'discrete'

        elif signalConstructorType == 'fromSingleThread':
            # constructorTuple = [SignalThread, SignalThread]
            self.independentThread = constructorTuple[0]
            self.dependentThread = constructorTuple[1]
            self.signalType = 'discrete'

        elif signalConstructorType == 'fromFunctionGenerator':
            # constructorTuple = [SignalThread, SignalThread]
            self.independentThread = constructorTuple[0]
            self.dependentThread = constructorTuple[1]
            self.signalType = 'continuous'

        elif signalConstructorType == 'fromSignalSequence':
            # constructorTuple = [SignalThread, SignalThread]
            self.independentThread = constructorTuple[0]
            self.dependentThread = constructorTuple[1]
            self.signalType = 'discrete'

    @classmethod
    def like(cls, signal:Signal):
        return cls('like', (np.zeros_like(signal.independentThread.data), signal.dependentThread.data))

    @classmethod
    def fromThreadPair(cls, independentThread:SignalThread, dependentThread:SignalThread):
        # Check independentThread type
        if isinstance(independentThread, SignalThread):
            independentThread = independentThread
        else:
            raise SignalTypeError('Independent signal thread must be of type "SignalThread".')

        # Check dependentThread type
        if isinstance(dependentThread, SignalThread):
            dependentThread = dependentThread
        else:
            raise SignalTypeError('Dependent signal thread must be of type "SignalThread".')

        # Check that the independent and dependent signal threads are the same length
        if independentThread.length != dependentThread.length:
            raise SignalValueError('Independent and Dependent signal threads must be the same length.')
            
        # Check that the dependent signal thread is strictly increasing (avoids problems with calculus operations)
        if not dependentThread.isIncreasing:
            raise SignalValueError('Dependent signal thread must be strictly increasing.')

        # Prepare constructor call
        signalConstructorType = 'fromThreadPair'
        return cls(signalConstructorType, (independentThread, dependentThread))

    @classmethod
    def fromSingleThread(cls, independentThread:SignalThread, parameterizationMethod:str, tStart:np.float64=0.0, totalArcLength:np.float64=None):
        # Check independentThread type
        if isinstance(independentThread, SignalThread):
            independentThread = independentThread
        else:
            raise SignalTypeError('Independent signal thread must be of type "SignalThread".')

        # Parameterize the independent thread
        if parameterizationMethod == 'indices':
            dependentThread = SignalThread(np.arange(0, independentThread.length, step=1))
        elif parameterizationMethod == 'arcLength':
            dependentThread = SignalThread(Signal.arcLength1D(independentThread.data, prependArcLength=tStart, totalArcLength=totalArcLength))
        else:
            raise SignalValueError('Parameterization method ''%s'' is not recognized.')

        # Prepare constructor call
        signalConstructorType = 'fromSingleThread'
        return cls(signalConstructorType, (independentThread, dependentThread))

    @classmethod
    def fromFunctionGenerator(cls, functionGenerator, parameterizationMethod=Tuple[str, Any]):
        # # Check function generator type
        # if not isinstance(functionGenerator, SignalSequence):
        #     raise SignalTypeError('functionGenerator must be of type "FunctionGenerator".')

        # Create a discrete version of the function
        if parameterizationMethod[0] == 'fromSignalThread':
            dependentThread = parameterizationMethod[1]
            independentThread = functionGenerator.toSignalThread(dependentThread)
        else:
            raise SignalTypeError('Unrecognized parameterization method.')

        # Prepare constructor call
        signalConstructorType = 'fromFunctionGenerator'
        return cls(signalConstructorType, (independentThread, dependentThread))

    @classmethod
    def fromSignalSequence(cls, signalList:List[Signal]) -> Signal:
        independentThread = SignalThread.fromThreadSequence(signal.independentThread for signal in signalList)
        dependentThread = SignalThread.fromThreadSequence(signal.dependentThread for signal in signalList)

        # Prepare constructor call
        signalConstructorType = 'fromSignalSequence'
        return cls(signalConstructorType, (independentThread, dependentThread))

    @staticmethod
    def arcLength1D(data:np.ndarray, prependArcLength:np.float64=0.0, totalArcLength:np.float64=None) -> np.ndarray:
        arcLength = np.cumsum(np.abs(np.ediff1d(data, to_begin=0.0)))
        if totalArcLength is not None:
            np.multiply(arcLength, totalArcLength / arcLength[-1], out=arcLength)
        np.add(arcLength, prependArcLength, out=arcLength)
        return arcLength

    def sample(self, tThread:SignalThread, interpolationMethod:Callable[[Signal, SignalThread], Signal]) -> Signal:
        return interpolationMethod(self, tThread)

class SignalBundle:
    def __init__(self):
        self.signals = {}

    @classmethod
    def fromSignalBundleSequence(cls, signalBundleList:List[SignalBundle]) -> SignalBundle:
        signalDict = {}
        for signalBundle in signalBundleList:
            for signalKey in signalBundle.signals.keys():
                if signalKey in signalDict:
                    signalDict[signalKey].append(signalBundle.signals[signalKey])
                else:
                    signalDict[signalKey] = [signalBundle.signals[signalKey]]

        compiledBundle = cls()
        for signalKey in signalDict.keys():
            compiledBundle.addSignal(signalKey, Signal.fromSignalSequence(signalDict[signalKey]))

        return compiledBundle

    @classmethod
    def fromSignalBundleArray(cls, signalBundleArray:np.ndarray, signalNames:List[str]) -> SignalBundle:
        if not (len(signalNames) == signalBundleArray.shape[0]-1):
            raise SignalValueError('Number of signal names must match the number of indpendent lines in signalBundleArray.')
        newSignalBundle = cls()
        for i, signalName in enumerate(signalNames):
            newSignalBundle.addSignal(signalName, Signal.fromThreadPair(SignalThread(signalBundleArray[i+1,:]), SignalThread(signalBundleArray[0,:])))

        return newSignalBundle

    # Computes the n-dimensional arclength along a parametric path defined by one or more Signals.
    # This function should be called using the output of SignalBundle.sample() to ensure that each
    # independent signal is sampled at the same times.
    @staticmethod
    def arcLengthND(dataBundleArray:np.ndarray, prependArcLength:np.float64=0.0, totalArcLength:np.float64=None, normalizeAxes=False) -> np.ndarray:
        # Normalize and scale each independent variable to the same range (-1,1)
        if normalizeAxes:
            normalizedArray = np.empty((dataBundleArray.shape[0]-1, dataBundleArray.shape[1]))
            for signalRow in range(0, dataBundleArray.shape[0]-1):
                normalizedArray[signalRow] = np.interp(dataBundleArray[signalRow+1], (dataBundleArray[signalRow+1].min(), dataBundleArray[signalRow+1].max()), (-1,1))
        # Calculate n-dim arc lengths and cumulative lengths
        np.cumsum(np.linalg.norm(np.diff(normalizedArray, axis=1), axis=0), out=dataBundleArray[0,1:])
        dataBundleArray[0,0] = 0.0
        # Normalize arc length by totalArcLength if given
        if totalArcLength is not None:
            np.multiply(dataBundleArray[0,:], totalArcLength / dataBundleArray[0,-1], out=dataBundleArray[0,:])
        # Offset arc length by prependArcLength if given
        np.add(dataBundleArray[0,:], prependArcLength, out=dataBundleArray[0,:])
        return dataBundleArray

    def addSignal(self, name:str, signal:Signal) -> None:
        if name in self.signals.keys():
            raise SignalValueError('Cannot add Signal (%s) to bundle. "%s" already exists.')

        self.signals[name] = signal

    def sample(self, tThread:SignalThread, signalInterpList:List[Tuple[str, Callable[[Signal, SignalThread], Signal]]]) -> np.ndarray:
        sampledSignalList = []
        for signalInterp in signalInterpList:
            if len(signalInterp) != 2:
                raise SignalValueError('Must provide Signal name and interpolation method.')
            if signalInterp[0] not in self.signals.keys():
                raise SignalValueError('Signal ''%s'' does not exist in this bundle.')

            sampledSignalList.append(self.signals[signalInterp[0]].sample(tThread, signalInterp[1]))

        return np.vstack((tThread.data, np.vstack([sampledSignal.independentThread.data for sampledSignal in sampledSignalList])))

#\todo - SignalSequence tThread is jenky...
class SignalSequence:
    def __init__(self, t0:np.float64=np.float64(0.0)):
        self.functionList = []
        self.duration = t0

    def appendFunction(self, function:Callable[[SignalThread], Signal], functionT0:np.float64, functionT1:np.float64) -> None:
        if functionT0 > functionT1:
            raise SignalValueError('functionT0 must be less than or equal to functionT1')

        self.functionList.append((function, functionT0, functionT1-functionT0))
        self.duration += abs(functionT1 - functionT0)

    # def evaluate(self, tThread:SignalThread) -> Signal:
    #     t = np.float64(0.0)
    #     if not tThread.isIncreasing:
    #         raise SignalValueError('tThread must be increasing.')

    #     signalList = []
    #     functionRegion = None
    #     for function in self.functionList:
    #         if function is self.functionList[-1]:
    #             functionRegion = np.where(np.logical_and(tThread.data >= t, tThread.data <= t+function[2]))[0]
    #         else:
    #             functionRegion = np.where(np.logical_and(tThread.data >= t, tThread.data < t+function[2]))[0]
    #         t += function[2]

    #         regionTThread = SignalThread(tThread.data[functionRegion[0]:functionRegion[-1]+1] - tThread.data[functionRegion[0]] + function[1])
    #         signalList.append(function[0](regionTThread))

    #     return Signal.fromSignalSequence(signalList, tThread)
