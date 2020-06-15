#!python3

import numpy as np
from magLabUtilities.signalutilities.signals import Signal, SignalThread
from magLabUtilities.signalutilities.functionGenerator import FunctionSequence, Line

if __name__=='__main__':
    up1 = Line(x0=0.0, x1=1.0, t0=0.0, t1=1.0)
    down1 = Line(x0=1.0, x1=-1.0, t0=0.0, t1=2.0)
    up2 = Line(x0=-1.0, x1=0.0, t0=0.0, t1=1.0)

    triangleCycle = FunctionSequence()

    triangleCycle.appendFunction(up1.evaluate, 0.0, 1.0)
    triangleCycle.appendFunction(down1.evaluate, 0.0, 2.0)
    triangleCycle.appendFunction(up2.evaluate, 0.0, 1.0)

    # triangleCycle.appendFunction(up1.evaluate, 0.0, 1.0)
    # triangleCycle.appendFunction(up1.evaluate, 1.0, 3.0)
    # triangleCycle.appendFunction(up1.evaluate, 3.0, 4.0)

    tThread = SignalThread(data=np.linspace(0.0, 4.0, 9))

    triangleSignal = triangleCycle.evaluate(tThread=tThread)

    print('done')
