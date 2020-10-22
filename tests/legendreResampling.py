#!python3

import numpy as np
from magLabUtilities.signalutilities.signals import SignalThread, Signal, SignalBundle
from magLabUtilities.signalutilities.hysteresis import HysteresisSignalBundle
from magLabUtilities.datafileutilities.timeDomain import importFromXlsx
from magLabUtilities.signalutilities.interpolation import Legendre
from magLabUtilities.uiutilities.plotting.hysteresis import MofHPlotter

if __name__ == '__main__':
    # # import data
    # fp = './tests/workflowTests/datafiles/CarlData.xlsx'
    # refBundle = HysteresisSignalBundle(importFromXlsx(fp, '22.7k', 1, 'A,B', dataColumnNames=['H','M']))

    # # Parameterize data by arc length
    # refBundleArray = np.vstack([refBundle.signals['H'].dependentThread.data, refBundle.signals['H'].independentThread.data, refBundle.signals['M'].independentThread.data])
    # refBundleArray = SignalBundle.arcLengthND(refBundleArray, totalArcLength=5.0)
    # refBundle = HysteresisSignalBundle.fromSignalBundleArray(refBundleArray, ['H', 'M'])

    # # Interpolate Data with roughly even arc lengths
    # tThread = SignalThread(np.linspace(0.0, 5.0, 500))
    # interpolator = Legendre(interpRadius=100, legendreOrder=5)
    # interpRefBundle = HysteresisSignalBundle()
    # interpRefBundle.addSignal('H', interpolator.interpolate(refBundle.signals['H'], tThread))
    # interpRefBundle.addSignal('M', interpolator.interpolate(refBundle.signals['M'], tThread))

    # # Plot stuff
    # plotter = MofHPlotter()
    # plotter.addPlot(refBundle, 'Ref')
    # plotter.addPlot(interpRefBundle, 'Interpolated')

    xThread = SignalThread(np.hstack([np.linspace(0.0, 100.0, 101), np.linspace(900.0, 1000.0, 101)]))
    yThread = SignalThread(np.hstack([np.linspace(0.0, 100.0, 101), np.linspace(900.0, 1000.0, 101)]))
    xSignal = Signal.fromSingleThread(xThread, 'indices', 0.0)
    ySignal = Signal.fromSingleThread(yThread, 'indices', 0.0)

    bundle = HysteresisSignalBundle()
    bundle.addSignal('X', xSignal)
    bundle.addSignal('Y', ySignal)

    bundleArray = np.vstack([bundle.signals['X'].dependentThread.data, bundle.signals['X'].independentThread.data, bundle.signals['Y'].independentThread.data])
    bundleArray = SignalBundle.arcLengthND(bundleArray, totalArcLength=1000.0)
    bundle = HysteresisSignalBundle.fromSignalBundleArray(bundleArray, ['X', 'Y'])

    tThread = SignalThread(np.linspace(0.0, 1000.0, 5))
    interpolator = Legendre(interpRadius=2, legendreOrder=1)
    interpBundle = HysteresisSignalBundle()
    interpBundle.addSignal('X', interpolator.interpolate(bundle.signals['X'], tThread))
    interpBundle.addSignal('Y', interpolator.interpolate(bundle.signals['Y'], tThread))


    print('done')
