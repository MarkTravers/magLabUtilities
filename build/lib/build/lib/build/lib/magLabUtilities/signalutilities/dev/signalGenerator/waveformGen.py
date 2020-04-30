#!python3

# \name
#   Waveform Generator module
#
# \description
#   Generates combinations of canonical waveforms
#
# \notes
#   parabola(self, xMaxStep, xMinStep, x0, x1, power, numPoints)
#       where: 
#           xMaxStep  - x-point with the largest step size (e.g. at saturation)
#           xMinStep  - x-point with the smallest step size (e.g. at coercive field)
#           power     - exponent on the parabola (fractional powers are acceptable)
#           x0        - beginning of the sweep
#           x1        - end of the sweep
#           numPoints - is the number of points in the sweep (inclusive)
#
#   sinusoid(self, amplitude, thetaStep, angularPeriod, numCycles, phaseShift=0.0)
#       where:
#           amplitude       - Amplitude of the sine wave
#           thetaStep       - step size of theta
#           angularPeriod   - angular period of sine wave (2pi is one cylce)
#           numCycles       - number of cycles the sine wave goes through
#           (phaseShift)    - shifts the sine wave (default is 0.0)
#
# \todo
#   implement these functions with the Signal class
#
# \revised
#   Mark Travers 4/22/2020      - Original construction
#   Stephen Gedney 4/27/2020    - Added Linear function

import numpy as np

class SignalGenerator:
    def __init__(self):
        self.signalList = []

    def appendSignal(self, signalPiece):
        if len(self.signalList) == 0:
            self.signalList.append(signalPiece)
        else:
            self.signalList.append(signalPiece[1:])

    def compileSignal(self):
        return np.hstack(self.signalList)

    def parabola(self, xMaxStep, xMinStep, x0, x1, power, numPoints):
        def mapToParabola(x, exponent):
            if (xMaxStep > xMinStep and xMinStep > x) or (xMaxStep < xMinStep and xMinStep < x):
                x = xMinStep + (xMinStep - x)
                return -(xMaxStep - xMinStep) * np.power((x-xMinStep)/(xMaxStep-xMinStep), exponent) + xMinStep
            else:
                return (xMaxStep - xMinStep) * np.power((x-xMinStep)/(xMaxStep-xMinStep), exponent) + xMinStep

        t0 = mapToParabola(x0, 1.0/power)
        t1 = mapToParabola(x1, 1.0/power)
        tArray = np.linspace(t0, t1, num=numPoints)
        mapArrayToParabola = np.vectorize(mapToParabola)
        xArray = mapArrayToParabola(tArray, power)
        return xArray

    def sinusoid(self, amplitude, thetaStep, angularPeriod, numCycles, phaseShift=0.0):
        numPoints = (angularPeriod * numCycles) / thetaStep
        tArray = np.linspace(0.0, numPoints*thetaStep, numPoints+1)
        return amplitude * np.sin(tArray-phaseShift)

    def linear(self, xStep, xStart, xStop):
        numPoints = np.floor(np.absolute((xStop - xStart) / xStep))
        npts = numPoints.astype(np.int)+1
        xArray = np.linspace(xStart, xStop, npts)
        return xArray

if __name__ == '__main__':
    # Create a signal generator
    sigGen = SignalGenerator()

    # ################ Minor loop example with parabolic point distribution #################
    # # Virgin curve
    # sigGen.appendSignal(sigGen.parabola(xMaxStep=10000.0, xMinStep=0.0, x0=0.0, x1=10000.0, power=1.5, numPoints=101))
    # # Positive reversal
    # sigGen.appendSignal(sigGen.parabola(xMaxStep=10000.0, xMinStep=-820.0, x0=10000.0, x1=-10000.0, power=1.5, numPoints=201))
    # # Negative reversal
    # sigGen.appendSignal(sigGen.parabola(xMaxStep=-10000.0, xMinStep=820.0, x0=-10000.0, x1=10000.0, power=1.5, numPoints=201))

    ################# Minor loop example with sinusoidal point distribution #################
    # sigGen.appendSignal(sigGen.sinusoid(amplitude=10000.0, thetaStep=np.pi/100, angularPeriod=2.0*np.pi, numCycles=1.25, phaseShift=0.0)) 

    # combine signals, excluding shared intermediary endpoints
    signal = sigGen.compileSignal()

    # calculate and display largest and smallest step sizes
    stepSizes = np.abs(signal[:-1] - signal[1:])
    print('Largest step size: %f' % np.amax(stepSizes))
    print('Smallest step size: %f' % np.amin(stepSizes))
    print('Mean step size: %f' % np.mean(stepSizes))
    print('Standard deviation of step size: %f' % np.std(stepSizes))

    # output to waveform file
    with open('./waveform.txt', 'w') as waveformFile:
        waveformFile.write('%s\n1.0\n' % signal.shape[0])
        for i in range(signal.shape[0]):
            waveformFile.write('%f\n' % signal[i])
