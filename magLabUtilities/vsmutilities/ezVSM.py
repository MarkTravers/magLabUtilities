#!python3

import numpy as np
from typing import List, Dict, Union
import warnings
from magLabUtilities.signalutilities.signals import SignalThread, Signal, SignalBundle
from magLabUtilities.config import Constants
from magLabUtilities.exceptions.exceptions import FileIOValueError

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
            # Time
            timeSinceMidnight=False,
            # Temperatures
            tempRawC=False,
            tempRawK=False,
            temp1C=False,
            temp1K=False,
            temp2C=False,
            temp2K=False,
            # H-applied
            hAppRawOe=False,
            hAppRawApm=False,
            hAppManipOe=False,
            hAppManipApm=False,
            hAppRawPlotOe=False,
            hAppRawPlotApm=False,
            hAppManipPlotOe=False,
            hAppManipPlotApm=False,
            # Angle of rotation stage
            fieldAngle=False,
            # Magnetization
            mXRawEmu=False,
            mXRawApm=False,
            mYRawEmu=False,
            mYRawApm=False,
            mXManipEmu=False,
            mXManipApm=False,
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
    # Extract start time from datafile
    startTime = extractStartTime(dataFileContents)

    ### Extract/convert data columns into data bundle array ###
    # Time
    tThread = SignalThread(dataFileArray[:,0])
    if timeSinceMidnight:
        dataThread = SignalThread(dataFileArray[:,0] + startTime)
        dataBundle.addSignal('timeSinceMidnight', Signal.fromThreadPair(dataThread, tThread))
    # Temperatures
    if tempRawC:
        dataThread = SignalThread(dataFileArray[:,1])
        dataBundle.addSignal('tempRawC', Signal.fromThreadPair(dataBundle, tThread))
    if tempRawK:
        dataThread = SignalThread(dataFileArray[:,1] + Constants.kToC)
        dataBundle.addSignal('tempRawK', Signal.fromThreadPair(dataBundle, tThread))
    if temp1C:
        dataThread = SignalThread(dataFileArray[:,2])
        dataBundle.addSignal('temp1C', Signal.fromThreadPair(dataBundle, tThread))
    if temp1K:
        dataThread = SignalThread(dataFileArray[:,2] + Constants.kToC)
        dataBundle.addSignal('temp1K', Signal.fromThreadPair(dataBundle, tThread))
    if temp2C:
        dataThread = SignalThread(dataFileArray[:,3])
        dataBundle.addSignal('temp2C', Signal.fromThreadPair(dataBundle, tThread))
    if temp2K:
        dataThread = SignalThread(dataFileArray[:,3] + Constants.kToC)
        dataBundle.addSignal('temp2K', Signal.fromThreadPair(dataBundle, tThread))
    # H-applied
    if hAppRawOe:
        dataThread = SignalThread(dataFileArray[:,4])
        dataBundle.addSignal('hAppRawOe', Signal.fromThreadPair(dataThread, tThread))
    if hAppRawApm:
        dataThread = SignalThread(dataFileArray[:,4] * Constants.oeToApm)
        dataBundle.addSignal('hAppRawApm', Signal.fromThreadPair(dataThread, tThread))
    if hAppManipOe:
        dataThread = SignalThread(dataFileArray[:,5])
        dataBundle.addSignal('hAppManipOe', Signal.fromThreadPair(dataThread, tThread))
    if hAppManipApm:
        dataThread = SignalThread(dataFileArray[:,5] * Constants.oeToApm)
        dataBundle.addSignal('hAppManipApm', Signal.fromThreadPair(dataThread, tThread))
    if hAppRawPlotOe:
        dataThread = SignalThread(dataFileArray[:,7])
        dataBundle.addSignal('hAppRawPlotOe', Signal.fromThreadPair(dataThread, tThread))
    if hAppRawPlotApm:
        dataThread = SignalThread(dataFileArray[:,7] * Constants.oeToApm)
        dataBundle.addSignal('hAppRawPlotApm', Signal.fromThreadPair(dataThread, tThread))
    if hAppManipPlotOe:
        dataThread = SignalThread(dataFileArray[:,8])
        dataBundle.addSignal('hAppManipPlotOe', Signal.fromThreadPair(dataThread, tThread))
    if hAppManipPlotApm:
        dataThread = SignalThread(dataFileArray[:,8] * Constants.oeToApm)
        dataBundle.addSignal('hAppManipPlotApm', Signal.fromThreadPair(dataThread, tThread))
    # Angle of rotation stage
    if fieldAngle:
        dataThread = SignalThread(dataFileArray[:,6])
        dataBundle.addSignal('fieldAngle', Signal.fromThreadPair(dataThread, tThread))
    # Magnetization
    mYActive = False
    if dataFileArray.shape[1] > 11:
        mYActive = True
    if mXRawEmu:
        dataThread = SignalThread(dataFileArray[:,9] / 1000.0)
        dataBundle.addSignal('mXRawEmu', Signal.fromThreadPair(dataThread, tThread))
    if mXRawApm:
        dataThread = SignalThread(dataFileArray[:,9] / (1000.0 * sampleDimensions['Volume']['value']))
        dataBundle.addSignal('mXRawApm', Signal.fromThreadPair(dataThread, tThread))
    if mYRawEmu:
        if not mYActive:
            raise FileIOValueError('The VSM file does not have M-y data, but ''mYRawEmu'' is requested.')
        dataThread = SignalThread(dataFileArray[:,10] / 1000.0)
        dataBundle.addSignal('mYRawEmu', Signal.fromThreadPair(dataThread, tThread))
    if mYRawApm:
        if not mYActive:
            raise FileIOValueError('The VSM file does not have M-y data, but ''mYRawApm'' is requested.')
        dataThread = SignalThread(dataFileArray[:,10] / (1000.0 * sampleDimensions['Volume']['value']))
        dataBundle.addSignal('mYRawApm', Signal.fromThreadPair(dataThread, tThread))
    if mXManipEmu:
        if mYActive:
            dataThread = SignalThread(dataFileArray[:,11] / 1000.0)
        else:
            dataThread = SignalThread(dataFileArray[:,10] / 1000.0)
        dataBundle.addSignal('mXManipEmu', Signal.fromThreadPair(dataThread, tThread))
    if mXManipApm:
        if mYActive:
            dataThread = SignalThread(dataFileArray[:,11] / (1000.0 * sampleDimensions['Volume']['value']))
        else:
            dataThread = SignalThread(dataFileArray[:,10] / (1000.0 * sampleDimensions['Volume']['value']))
        dataBundle.addSignal('mXManipApm', Signal.fromThreadPair(dataThread, tThread))
    if mYManipEmu:
        if not mYActive:
            raise FileIOValueError('The VSM file does not have M-y data, but ''mYManipEmu'' is requested.')
        dataThread = SignalThread(dataFileArray[:,10] / 1000.0)
        dataBundle.addSignal('mYManipEmu', Signal.fromThreadPair(dataThread, tThread))
    if mYManipApm:
        if not mYActive:
            raise FileIOValueError('The VSM file does not have M-y data, but ''mYManipApm'' is requested.')
        dataThread = SignalThread(dataFileArray[:,12] / (1000.0 * sampleDimensions['Volume']['value']))
        dataBundle.addSignal('mYManipApm', Signal.fromThreadPair(dataThread, tThread))

    return dataBundle

def extractStartTime(dataFileContents:List[str]) -> np.float64:
    for line in dataFileContents:
        if '@Time at start of measurement: ' in line:
            return np.float64(3600 * np.int32(line[-9:-7]) + 60 * np.int32(line[-6:-4]) + np.int32(line[-3:-1]))

def extractSampleDimensions(dataFileContents:List[str]) -> Dict[str,Dict[str,Union[float,str]]]:
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
                sampleDimensionDict[dimension[0]]['value'] = np.float64(temp[0])
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


