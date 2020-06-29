#!python3

import numpy as np
from typing import Tuple, List
import matlab.engine
from magLabUtilities.signalutilities.hysteresis import HysteresisSignalBundle
from magLabUtilities.exceptions.exceptions import UITypeError, UIValueError

class MofHPlotter:
    def __init__(self, hysteresisBundleList:List[Tuple[HysteresisSignalBundle, str]]=[]):
        self.hysteresisBundleList = HysteresisSignalBundle

        if not isinstance(hysteresisBundleList, list):
            raise UIValueError('hysteresisBundleList must be of type ''List''.')

        self.matEng = matlab.engine.start_matlab()
        self.fig = self.matEng.figure()
        figPosition = [300, 300, 700, 600]
        self.matEng.set(self.matEng.gcf(), 'position', matlab.double(figPosition))
        self.matEng.subplot('111')
        self.matEng.sgtitle('M(H)')

        for hysteresisBundle in hysteresisBundleList:
            self.addPlot(*hysteresisBundle)

    def addPlot(self, hysteresisBundle:HysteresisSignalBundle, plotName:str):
        if not isinstance(hysteresisBundle, HysteresisSignalBundle):
            raise UITypeError('Invalid argument for addPlot()')
        
        if not hysteresisBundle.mhComplete:
            raise UIValueError('hysteresisBundle is not mh complete.')

        plotX = matlab.double(hysteresisBundle.signals['H'].independentThread.data.tolist())
        plotY = matlab.double(hysteresisBundle.signals['M'].independentThread.data.tolist())

        self.matEng.plot(plotX, plotY, 'DisplayName', plotName)
        self.matEng.hold('on', nargout=0)
        self.matEng.grid('on', nargout=0)
        self.matEng.xlabel('Total Field [A/m]')
        self.matEng.ylabel('Magnetization [A/m]')
        self.matEng.legend('location', 'northwest')

class XofMPlotter:
    def __init__(self, hysteresisBundleList:List[Tuple[HysteresisSignalBundle, str]]=[]):
        self.hysteresisBundleList = HysteresisSignalBundle

        if not isinstance(hysteresisBundleList, list):
            raise UIValueError('hysteresisBundleList must be of type ''List''.')

        self.matEng = matlab.engine.start_matlab()
        self.fig = self.matEng.figure()
        figPosition = [300, 300, 700, 600]
        self.matEng.set(self.matEng.gcf(), 'position', matlab.double(figPosition))
        self.matEng.subplot('111')
        self.matEng.sgtitle('\\chi(M)')

        for hysteresisBundle in hysteresisBundleList:
            self.addPlot(*hysteresisBundle)

    def addPlot(self, hysteresisBundle:HysteresisSignalBundle, plotName:str):
        if not isinstance(hysteresisBundle, HysteresisSignalBundle):
            raise UITypeError('Invalid argument for addPlot()')
        
        if not hysteresisBundle.xmComplete:
            raise UIValueError('hysteresisBundle is not xm complete.')

        plotX = matlab.double(hysteresisBundle.signals['M'].independentThread.data.tolist())
        plotY = matlab.double(hysteresisBundle.signals['X'].independentThread.data.tolist())

        self.matEng.semilogy(plotX, plotY, 'DisplayName', plotName)
        self.matEng.hold('on', nargout=0)
        self.matEng.grid('on', nargout=0)
        self.matEng.xlabel('Magnetization [A/m]')
        self.matEng.ylabel('Susceptibility')
        self.matEng.legend('location', 'south')

# This section uses matplotlib to plot things.
# \todo - Autodetect presence of Matlab on system. Either that or make the user choose matplotlib or matlab.
# import matplotlib.pyplot as plt
# class MofHPlotter:
#     def __init__(self, hysteresisBundleList:List[Tuple[HysteresisSignalBundle, str]]=[]):
#         self.hysteresisBundleList = hysteresisBundleList

#         plt.ion()
#         plt.show()
#         self.fig = plt.figure(figsize=(10,7))
#         self.ax = self.fig.add_subplot(1,1,1)

#         if not isinstance(hysteresisBundleList, list):
#             raise UIValueError('hysteresisBundleList must be of type ''List''.')

#         for hysteresisBundle in hysteresisBundleList:
#             self.addPlot(*hysteresisBundle)

#     def addPlot(self, hysteresisBundle:HysteresisSignalBundle, plotName:str):
#         if not isinstance(hysteresisBundle, HysteresisSignalBundle):
#             raise UITypeError('Invalid argument for addPlot()')
        
#         if not hysteresisBundle.mhComplete:
#             raise UIValueError('hysteresisBundle is not mh complete.')

#         self.ax.plot(hysteresisBundle.signals['H'].independentThread.data, hysteresisBundle.signals['M'].independentThread.data, label=plotName)
#         self.ax.set_xlabel('H [A/m]')
#         self.ax.set_ylabel('M [A/m]')
#         self.ax.grid(True)
#         self.ax.legend(fontsize=10)

#         self.fig.tight_layout()
#         plt.draw()
#         plt.pause(0.001)
