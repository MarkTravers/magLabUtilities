#!python3

import numpy as np
import pandas as pd
import re
from typing import List
from magLabUtilities.signalutilities.signals import SignalThread, Signal, SignalBundle
from magLabUtilities.exceptions.exceptions import FileIOValueError

def importFromXlsx(fp:str, sheetName:str, headerRow:int, excelDataColumns:str, excelTColumn:str=None, dataColumnNames:List[str]=None) -> SignalBundle:
        dataColumns = pd.read_excel(fp, sheet_name=sheetName, header=headerRow-1, usecols=excelDataColumns, dtype=np.float64)

        if excelTColumn is not None:
            if not excelTColumn.isalpha():
                raise FileIOValueError('Only one tColumn supported.')
            tColumn = pd.read_excel(fp, sheet_name=sheetName, header=None, skiprows=headerRow, usecols=excelTColumn, dtype=np.float64)

        if dataColumnNames is not None:
            if len(dataColumnNames) != len(dataColumns.columns):
                raise FileIOValueError('Must specify the same number of column names as number of columns requested.')

        fileSignalBundle = SignalBundle()

        for i, dataColumnKey in enumerate(dataColumns.columns):
            dataColumn = dataColumns[dataColumnKey].dropna()
            independentThread = SignalThread(dataColumn.values)
            if excelTColumn is None:
                dependentThread = SignalThread(dataColumn.index.values)
            else:
                dependentThread = SignalThread(tColumn.values[dataColumn.index.values].T[0])

            if dataColumnNames is None:
                fileSignalBundle.addSignal(dataColumnKey, Signal.fromThreadPair(independentThread, dependentThread))
            else:
                fileSignalBundle.addSignal(dataColumnNames[i], Signal.fromThreadPair(independentThread, dependentThread))

        return fileSignalBundle

def exportSignalArrayToXlsx(fp:str, exportBundleArray:np.ndarray, columnNames:List[str], sheetName:str):
    arrayDataFrame = pd.DataFrame(exportBundleArray.T)
    arrayDataFrame.to_excel(fp, sheet_name=sheetName, header=columnNames, index=False)
