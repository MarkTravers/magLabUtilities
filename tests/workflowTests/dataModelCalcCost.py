#!python3

import numpy as np
from magLabUtilities.datafileutilities.timeDomain import importFromXlsx
from magLabUtilities.signalutilities.signals import SignalThread, Signal, SignalBundle
from magLabUtilities.signalutilities.hysteresis import XExpQA, HysteresisSignalBundle
from magLabUtilities.uiutilities.plotting.hysteresis import MofHPlotter, XofMPlotter

if __name__=='__main__':
    # fp = './tests/workflowTests/datafiles/M(H)_Curves_Netza.xlsx'
    # refBundle = HysteresisSignalBundle(importFromXlsx(fp, '21k', 2, 'C,D', dataColumnNames=['M','H']))

    # xExpQA = XExpQA(xInit=67.0, hCoercive=630.0, mSat=1.67e6, hCoop=1200.0, hAnh=2000.0, xcPow=4.0, mRev=-11.0)
    # testBundle = HysteresisSignalBundle(xExpQA.evaluate(refBundle.signals['M']))

    # mhPlotter = MofHPlotter([(refBundle, 'Ref0')])

    # xmPlotter = XofMPlotter([(testBundle, 'Test0')])

    print('done')
