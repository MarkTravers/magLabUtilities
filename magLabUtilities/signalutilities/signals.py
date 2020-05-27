#!python3
import numpy as np
from magLabUtilities.exceptions.exceptions import SignalTypeError, SignalValueError
from magLabUtilities.signalutilities.functions import FunctionGenerator

class SignalThread:
    def __init__(self, data):
        # Convert various input datatypes to a Numpy array
        if isinstance(data, np.ndarray):
            self.data = data

        elif isinstance(data, Signal):
            self.data = data.data

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
    def __init__(self, independentThread=None, dependentThread=None, functionGenerator=None):
        if isinstance(functionGenerator, FunctionGenerator):
            self.functionGenerator = functionGenerator
        # Check to make sure that the independent thread is a SignalThread instance
        if isinstance(independentThread, SignalThread):
            self.independentThread = independentThread
        else:
            raise SignalTypeError('Dependent signal thread must be of type "SignalThread".')

        # Check to make sure that the dependent thread is a SignalThread instance
        if isinstance(dependentThread, SignalThread):
            self.dependentThread = dependentThread

            # Check that the independent and dependent signal threads are the same length
            if self.independentThread.length != self.dependentThread.length:
                raise SignalValueError('Independent and Dependent signal threads must be the same length.')
                
            # Check that the dependent signal thread is strictly increasing (avoids problems with calculus operations)
            if not dependentThread.isIncreasing:
                raise SignalValueError('Dependent signal thread must be strictly increasing.')

        elif dependentThread == 'indices':
            # \TODO implement this warning with message pipe once message pipe is converted to static functions.
            print('Signal has been linearly parameterized along its indices.')
            self.dependentThread = np.arange(0, self.independentThread.length, step=1)

        else:
            raise SignalTypeError('Dependent thread must be of type "SignalThread" or an automatic constructor method specified as type "str".')

    def generateInterpolationFunction(self, interpolationMethod, methodParameters):
        if interpolationMethod == 'legendre':
            pass

        else:
            raise SignalValueError('No interpolation method: %s' % interpolationMethod)
