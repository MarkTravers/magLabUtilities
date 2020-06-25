#!python3

import numpy as np
from magLabUtilities.signalutilities.signals import SignalThread, Signal
from magLabUtilities.signalutilities.canonical1d import Parabola

if __name__=='__main__':
    parabolaGen = Parabola(tMaxDx=4.0, tVertex=0.0, xMaxDx=16.0, xVertex=0.0, power=2.0)
    parabola = parabolaGen.evaluate(SignalThread(np.linspace(0.0, 4.0, num=5)))

    interpTThread = SignalThread(np.array([0.0, 2.0, 4.0]))

    interpolatedSignal = parabola.sample(interpTThread, 'nearestPoint')

    print('done')