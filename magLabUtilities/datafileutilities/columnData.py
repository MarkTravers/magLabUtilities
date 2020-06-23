#!python3

import numpy as np
import pandas as pd
from typing import List
from magLabUtilities.signalutilities.signals import SignalThread, Signal, SignalBundle

class importFromXlsx:
    @staticmethod
    def columnHeadersFromFile(fp:str, sheetName:str, headerRow:int, dataColumns:str, tColumn:str=None):
        fileData = pd.read_excel(fp, sheet_name=sheetName, header=headerRow-1, usecols=dataColumns, dtype=np.float64)
        fileSignalBundle = SignalBundle()
        for columnkey in fileData:
            

    @staticmethod
    def specifyColumnHeaders(fp:str, sheetName:str, dataStartRow:int, dataColumns:str, columnHeaders:List[str]):
