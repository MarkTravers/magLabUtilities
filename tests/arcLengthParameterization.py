#!python3

import numpy as np
from magLabUtilities.datafileutilities.timeDomain import importFromXlsx
from magLabUtilities.signalutilities.signals import Signal, SignalBundle, SignalThread
from magLabUtilities.signalutilities.interpolation import nearestPoint
from magLabUtilities.uiutilities.plotting.hysteresis import HysteresisSignalBundle, MofHPlotter

if __name__ == '__main__':
    # # Creat a circle and parameterize it by arclength
    # plotter = MofHPlotter()

    # tThread = SignalThread(np.linspace(0, 2.0*np.pi, num=100))
    # hThread = SignalThread(np.cos(tThread.data))
    # mThread = SignalThread(np.sin(tThread.data))

    # hSignal = Signal.fromThreadPair(hThread, tThread)
    # mSignal = Signal.fromThreadPair(mThread, tThread)

    # circleBundle = SignalBundle()
    # circleBundle.addSignal('H', hSignal)
    # circleBundle.addSignal('M', mSignal)

    # circleBundleArray = circleBundle.sample(SignalThread([0.0, np.pi/2.0, np.pi, 3.0*np.pi/2.0, 2.0*np.pi]), [('H', nearestPoint), ('M', nearestPoint)])
    # arcLenParamCircleBundle = SignalBundle.fromSignalBundleArray(circleBundle.arcLengthND(circleBundleArray), ['H', 'M'])

    # plotter.addPlot(HysteresisSignalBundle(arcLenParamCircleBundle), '')

    # Import hysteresis data and parameterize it by arclength
    fp = './tests/workflowTests/datafiles/CarlData.xlsx'
    mhRawBundle = importFromXlsx(fp, '22.7k', 1, 'A,B')
    mhRawArray = mhRawBundle.sample(mhRawBundle.signals['M'].dependentThread, [('M', nearestPoint), ('H', nearestPoint)])
    mhParamBundle = SignalBundle.fromSignalBundleArray(mhRawBundle.arcLengthND(mhRawArray, totalArcLength=1.0, normalizeAxes=True), ['M', 'H'])

    mhResampleArray = mhParamBundle.sample(SignalThread(np.linspace(0.0, 1.0, 30)), [('M', nearestPoint), ('H', nearestPoint)])
    mhResampleBundle = SignalBundle.fromSignalBundleArray(mhResampleArray, ['M', 'H'])

    plotter = MofHPlotter()
    plotter.addPlot(HysteresisSignalBundle(mhRawBundle), 'Raw')
    plotter.addPlot(HysteresisSignalBundle(mhResampleBundle), 'Resampled')

    print('done')
