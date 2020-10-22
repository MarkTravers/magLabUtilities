#!python3

import numpy as np
from magLabUtilities.signalutilities.signals import SignalThread, Signal
from magLabUtilities.signalutilities.calculus import integralTrapQuadrature

if __name__ == '__main__':
    x = SignalThread(np.array([1,2,4]))
    t = SignalThread(np.array([0,1,3]))
    intXDx = integralTrapQuadrature(x, t)

    print('here')
