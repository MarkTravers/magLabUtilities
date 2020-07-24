#!python3

import numpy as np
from typing import Tuple, List, Any
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

        for hysteresisBundle in hysteresisBundleList:
            self.addPlot(*hysteresisBundle)

    def addPlot(self, hysteresisBundle:HysteresisSignalBundle, plotName:str):
        self.plotMofH(self.matEng, hysteresisBundle, plotName)

    @staticmethod
    def plotMofH(matlabEngine:matlab.engine.matlabengine.MatlabEngine, mhBundle:HysteresisSignalBundle, plotName:str) -> None:
        if not isinstance(mhBundle, HysteresisSignalBundle):
            raise UITypeError('Invalid argument for addPlot()')
        
        if not mhBundle.mhComplete:
            raise UIValueError('hysteresisBundle is not mh complete.')

        plotX = matlab.double(mhBundle.signals['H'].independentThread.data.tolist())
        plotY = matlab.double(mhBundle.signals['M'].independentThread.data.tolist())

        matlabEngine.plot(plotX, plotY, 'DisplayName', plotName)
        matlabEngine.hold('on', nargout=0)
        matlabEngine.grid('on', nargout=0)
        matlabEngine.title('M(H)')
        matlabEngine.xlabel('Total Field [A/m]')
        matlabEngine.ylabel('Magnetization [A/m]')
        # matlabEngine.legend('location', 'southeast')
        # matlabEngine.hold('off', nargout=0)

class XofMPlotter:
    def __init__(self, hysteresisBundleList:List[Tuple[HysteresisSignalBundle, str]]=[]):
        self.hysteresisBundleList = HysteresisSignalBundle

        if not isinstance(hysteresisBundleList, list):
            raise UIValueError('hysteresisBundleList must be of type ''List''.')

        self.matEng = matlab.engine.start_matlab()
        self.fig = self.matEng.figure()
        figPosition = [300, 300, 600, 600]
        self.matEng.set(self.matEng.gcf(), 'position', matlab.double(figPosition))
        self.matEng.subplot('111')

        for hysteresisBundle in hysteresisBundleList:
            self.addPlot(*hysteresisBundle)

    def addPlot(self, hysteresisBundle:HysteresisSignalBundle, plotName:str):
        self.plotXofM(self.matEng, hysteresisBundle, plotName)

    @staticmethod
    def plotXofM(matlabEngine:matlab.engine.matlabengine.MatlabEngine, xmBundle:HysteresisSignalBundle, plotName:str):
        if not isinstance(xmBundle, HysteresisSignalBundle):
            raise UITypeError('Invalid argument for addPlot()')
        
        if not xmBundle.xmComplete:
            raise UIValueError('hysteresisBundle is not xm complete.')

        plotX = matlab.double(xmBundle.signals['M'].independentThread.data.tolist())
        plotY = matlab.double(xmBundle.signals['X'].independentThread.data.tolist())

        matlabEngine.semilogy(plotX, plotY, 'DisplayName', plotName)
        # matlabEngine.plot(plotX, plotY, 'DisplayName', plotName)
        matlabEngine.hold('on', nargout=0)
        matlabEngine.grid('on', nargout=0)
        matlabEngine.title('\\chi(M)')
        matlabEngine.xlabel('Magnetization [A/m]')
        matlabEngine.ylabel('Susceptibility')
        # matlabEngine.legend('location', 'south')
        # matlabEngine.hold('off', nargout=0)

    @staticmethod
    def plotXRevofM(matlabEngine:matlab.engine.matlabengine.MatlabEngine, xRevmBundle:HysteresisSignalBundle, plotName:str):
        if not isinstance(xRevmBundle, HysteresisSignalBundle):
            raise UITypeError('Invalid argument for addPlot()')
        
        if not xRevmBundle.xRevmComplete:
            raise UIValueError('hysteresisBundle is not xRevm complete.')

        plotX = matlab.double(xRevmBundle.signals['M'].independentThread.data.tolist())
        plotY = matlab.double(xRevmBundle.signals['Xrev'].independentThread.data.tolist())

        matlabEngine.semilogy(plotX, plotY, 'DisplayName', plotName)
        # matlabEngine.plot(plotX, plotY, 'DisplayName', plotName)
        matlabEngine.hold('on', nargout=0)
        matlabEngine.grid('on', nargout=0)
        matlabEngine.title('\\chi(M)')
        matlabEngine.xlabel('Magnetization [A/m]')
        matlabEngine.ylabel('Susceptibility')
        # matlabEngine.legend('location', 'south')
        # matlabEngine.hold('off', nargout=0)

class MofHXofMPlotter:
    def __init__(self):
        self.matEng = matlab.engine.start_matlab()
        self.fig = self.matEng.figure()
        figPosition = [100, 300, 1500, 600]
        self.matEng.set(self.matEng.gcf(), 'position', matlab.double(figPosition))
    
    def addMofHPlot(self, hysteresisBundle:HysteresisSignalBundle, plotName:str):
        self.matEng.subplot('121')
        MofHPlotter.plotMofH(self.matEng, hysteresisBundle, plotName)

    def addXofMPlot(self, hysteresisBundle:HysteresisSignalBundle, plotName:str):
        self.matEng.subplot('122')
        XofMPlotter.plotXofM(self.matEng, hysteresisBundle, plotName)

    def addXRevofMPlot(self, hysteresisBundle:HysteresisSignalBundle, plotName:str):
        self.matEng.subplot('122')
        XofMPlotter.plotXRevofM(self.matEng, hysteresisBundle, plotName)

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
