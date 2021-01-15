#!python

from magLabUtilities.vsmutilities.ezVSM import importDataFile
from magLabUtilities.uiutilities.plotting.hysteresis import MofHPlotter
from magLabUtilities.signalutilities.hysteresis import HysteresisSignalBundle

if __name__=='__main__':
    # Import vsm datafile
    datafileFP = './tests/fileReaderTests/010501_01-12-2021.RAW'
    vsmDataBundle = HysteresisSignalBundle(importDataFile(datafileFP, timeSinceMidnight=True, hAppRawApm=True, mXRawApm=True))

    # Rename signals for plotter
    vsmDataBundle.signals['M'] = vsmDataBundle.signals['mXRawApm']
    vsmDataBundle.signals['H'] = vsmDataBundle.signals['hAppRawApm']
    # Plot Hysteresis
    plotter = MofHPlotter()
    plotter.addPlot(vsmDataBundle, 'test')

    print('done')