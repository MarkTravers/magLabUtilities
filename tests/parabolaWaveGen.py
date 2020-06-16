#!python3

import numpy as np
from magLabUtilities.signalutilities.signals import Signal, SignalThread
from magLabUtilities.signalutilities.functionGenerator import FunctionSequence, Parabola

if __name__=='__main__':
    up = Parabola(tMaxDx=2.0, tVertex=0.0, xMaxDx=4.0, xVertex=0.0, power=2.0)
    down = Parabola(tMaxDx=-2.0, tVertex=0.0, xMaxDx=4.0, xVertex=0.0, power=2.0)

    vsmMInputSequence = FunctionSequence(t0=0.0)

    vsmMInputSequence.appendFunction(function=up.evaluate, functionT0=-2.0, functionT1=2.0)
    vsmMInputSequence.appendFunction(function=down.evaluate, functionT0=-2.0, functionT1=2.0)

    rawMInput = SignalThread(np.linspace(0.0, 8.0, 9))

    vsmInput = vsmMInputSequence.evaluate(rawMInput)

    print('done')
