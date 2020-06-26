#!python3
import numpy as np
from typing import List, Tuple, Any
from magLabUtilities.exceptions.exceptions import SignalTypeError, SignalValueError

class SignalThread:
    def __init__(self, data):
        # Convert various input datatypes to a Numpy array
        if isinstance(data, np.ndarray):
            self.data = data

        elif isinstance(data, list):
            self.data = np.asarray(self.data)

        else:
            raise SignalTypeError('Cannot create signal from supplied data type.')

        # Ensure that the Numpy array is 1-dimensional
        if len(data.shape) > 1:
            raise SignalTypeError('Cannot create signal from multi-dimensional array.')

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
        if signalConstructorType == 'fromThreadPair':
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
    def fromSingleThread(cls, independentThread:SignalThread, parameterizationMethod:str):
        # Check independentThread type
        if isinstance(independentThread, SignalThread):
            independentThread = independentThread
        else:
            raise SignalTypeError('Independent signal thread must be of type "SignalThread".')

        # Parameterize the independent thread
        if parameterizationMethod == 'indices':
            dependentThread = SignalThread(np.arange(0, independentThread.length, step=1))

        # Prepare constructor call
        signalConstructorType = 'fromSingleThread'
        return cls(signalConstructorType, (independentThread, dependentThread))

    @classmethod
    def fromFunctionGenerator(cls, functionGenerator, parameterizationMethod=Tuple[str, Any]):
        # Check function generator type
        from magLabUtilities.signalutilities.canonical1d import FunctionSequence
        if not isinstance(functionGenerator, FunctionSequence):
            raise SignalTypeError('functionGenerator must be of type "FunctionGenerator".')

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
    def fromSignalSequence(cls, signalList, sequenceDependantThread):
        independentThread = SignalThread(np.hstack((signal.independentThread.data for signal in signalList)))

        # Prepare constructor call
        signalConstructorType = 'fromSignalSequence'
        return cls(signalConstructorType, (independentThread, sequenceDependantThread))

    def generateInterpolationFunction(self, interpolationMethod, methodParameters):
        if interpolationMethod == 'legendre':
            pass

        else:
            raise SignalValueError('No interpolation method: %s' % interpolationMethod)

    def sample(self, tThread:SignalThread, interpolationMethod:str):
        if interpolationMethod == 'nearestPoint':
            from magLabUtilities.signalutilities.interpolation import nearestPoint
            return nearestPoint(self, tThread)
        else:
            raise SignalValueError('No interpolation method ''%s''.' % interpolationMethod)

class SignalBundle:
    def __init__(self):
        self.signals = {}

    def addSignal(self, name:str, signal:Signal, overrideCompilation=False) -> None:
        if name in self.signals.keys():
            raise SignalValueError('Cannot add Signal (%s) to bundle. "%s" already exists.')

        self.signals[name] = signal

    def sample(self, tThread:SignalThread, signalInterpList:List[Tuple[str, str]]) -> np.ndarray:
        sampledSignalList = []
        for signalInterp in signalInterpList:
            if len(signalInterp) != 2:
                raise SignalValueError('Must provide Signal name and interpolation method.')
            if signalInterp[0] not in self.signals.keys():
                raise SignalValueError('Signal ''%s'' does not exist in this bundle.')

            sampledSignalList.append(self.signals[signalInterp[0]].sample(tThread, signalInterp[1]))

        return np.vstack((tThread.data, np.vstack((sampledSignal.independentThread.data for sampledSignal in sampledSignalList))))
