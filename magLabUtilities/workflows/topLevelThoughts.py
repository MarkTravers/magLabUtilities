#!python3

from magstrommanager.testCase import MagstromTestCase
from magstrommanager.files import MagstromFiles

from optimizers import GradientDescent
from optimizers.utilities import TestManager, ParameterSpace, LossFunction

from dsp.utilities import DspChain, DataFrame
from dsp.calculus import integral, derivative
from dsp.interpolation import legendreInterpolation
from dsp.conditioners import parameterize, weightRegions, scale

from datafileutilities import decoders, encoders



