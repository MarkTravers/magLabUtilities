#!python3

import numpy as np
from magLabUtilities.signalutilities.signals import SignalThread, Signal, SignalBundle
from magLabUtilities.datafileutilities.columnData import importFromXlsx

if __name__ == '__main__':
    fp = './tests/fileReaderTests/M(H)_Curve.xlsx'
    fileSignalBundle = importFromXlsx(fp, '21k', 3, 'C,D,E', excelTColumn='B', dataColumnNames=['Magnetization', 'Magnetic Field', 'Susceptibility'])

    tThread = SignalThread(np.array([0.0, 1.44e17, 3.17e29]))
    sampleMatrix = fileSignalBundle.sample(tThread, [('Magnetization','nearestPoint'),('Magnetic Field', 'nearestPoint'),('Susceptibility', 'nearestPoint')])

    print('done')
