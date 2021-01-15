#!python3

import numpy as np
from typing import List, Dict, Union
from magLabUtilities.signalutilities.signals import SignalThread, Signal, SignalBundle

# Column 0: Time since start, Time [s]
# Column 1: Raw Temperature, Sample Temperature [degC]
# Column 2: Temperature, Sample Temperature [degC]
# Column 3: Temperature 2, Sample Temperature [degC]
# Column 4: Raw Applied Field, Applied Field [Oe]
# Column 5: Applied Field, Applied Field [Oe]
# Column 6: Field Angle, Field Angle [deg]
# Column 7: Raw Applied Field For Plot , Applied Field [Oe]
# Column 8: Applied Field For Plot , Applied Field [Oe]
# Column 9: Raw Signal Mx, Moment as measured [memu]
# Column 10: Signal X direction, Moment [emu]

# Column 0: Time since start, Time [s]
# Column 1: Raw Temperature, Sample Temperature [degC]
# Column 2: Temperature, Sample Temperature [degC]
# Column 3: Temperature 2, Sample Temperature [degC]
# Column 4: Raw Applied Field, Applied Field [Oe]
# Column 5: Applied Field, Applied Field [Oe]
# Column 6: Field Angle, Field Angle [deg]
# Column 7: Raw Applied Field For Plot , Applied Field [Oe]
# Column 8: Applied Field For Plot , Applied Field [Oe]
# Column 9: Raw Signal Mx, Moment as measured [memu]
# Column 10: Raw Signal My, Moment as measured [memu]
# Column 11: Signal X direction, Moment [emu]
# Column 12: Signal Y direction, Moment [emu]
def importDataFile(dataFileFP, 
            timeSinceStart=False,
            timeSinceMidnight=False,
            temp1C=False,
            temp1K=False,
            temp2C=False,
            temp2K=False,
            hAppRawOe=False,
            hAppRawApm=False,
            hAppManipOe=False,
            hAppManipApm=False,
            fieldAngle=False,
            hAppRawPlotOe=False,
            hAppRawPlotApm=False,
            hAppManipPlotOe=False,
            hAppManipPlotApm=False,
            mXRawMEmu=False,
            mXRawEmu=False,
            mXRawApm=False,
            mYRawMEmu=False,
            mYRawEmu=False,
            mYRawApm=False,
            mXManipMEmu=False,
            mXManipEmu=False,
            mXManipApm=False,
            mYManipMEmu=False,
            mYManipEmu=False,
            mYManipApm=False,
        ) -> SignalBundle:
    # Initialize SignalBundle for VSM data
    dataBundle = SignalBundle()
    # Read in VSM data file
    with open(dataFileFP) as vsmDataFile:
        dataFileContents = vsmDataFile.readlines()
    # Extract data table from vsm datafile
    dataFileArray = extractDataTable(dataFileContents)
    # Extract sample dimensions from datafile
    sampleDimensions = extractSampleDimensions(dataFileContents)

    # Extract/convert data columns into data bundle
    if timeSinceStart:
        tThread = SignalThread(dataFileArray[:,0])

def extractSampleDimensions(dataFileContents:List[str]) -> Dict[str:Dict[str:Union[float,str]]]:
    sampleDimensionLine = ''
    for n, line in enumerate(dataFileContents):
        if '@@Sample Dimensions' in line:
            sampleDimensionLine = dataFileContents[n+1]
            break

    # Shape = Cylindrical;  Length = 0.20 [mm] Width = 6.60 [mm] Thickness = 1.000E+3 [nm] Diameter = 26.07 [mm] Volume : 9.656E-8 [m^3] Area = 0.000E+0 [mm^2] Mass = 7.551E-1 [g] Nd =  0.00 Sample Angle Offset = 0.000 
    # volume is by mass and density
    sampleDimensionDict = {}
    sampleDimensionLine = sampleDimensionLine.replace('  ', ' ')
    sampleDimensionLine = sampleDimensionLine.replace(' = ', '=')
    sampleDimensionLine = sampleDimensionLine.replace(' : ', '=')
    sampleDimensionLine = sampleDimensionLine.replace(' [', '![')

    sampleDimensions = sampleDimensionLine.split(' ')
    for sampleDimension in sampleDimensions:
        if '=' in sampleDimension:
            dimension = sampleDimension.split('=')
            sampleDimensionDict[dimension[0]] = {}
            if '!' in dimension[1]:
                temp = dimension[1].split('!')
                sampleDimensionDict[dimension[0]]['value'] = temp[0]
                sampleDimensionDict[dimension[0]]['unit'] = temp[1]
            else:
                sampleDimensionDict[dimension[0]]['parameter'] = dimension[1]
    return(sampleDimensionDict)

def extractDataTable(dataFileContents:List[str]) -> np.ndarray:
    startLine = None
    stopLine = None
    for n in range(len(dataFileContents)):
        if '@@Data' in dataFileContents[n]:
            # Skip two lines to skip the header
            startLine = n + 2
        if '@@END Data.' in dataFileContents[n]:
            # Back up a line to skip the footer
            stopLine = n
            break
    return(np.genfromtxt(dataFileContents[startLine:stopLine], dtype=np.float64))
