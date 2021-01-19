#!python

from typing import Dict, List, Union, Callable, Any
from magLabUtilities.optimizerutilities.parameterSpaces import ParameterSpace

class FitKernel:
    def __init__(
                    self,
                    refFP:str,
                    parameterDefs:Dict[str, Dict[str, Any]], 
                    optimizerConstructorMethod:Callable[...,Callable],
                    optimizerConfig:Dict[str, Dict[str, Any]],
                    tuneHistoryFP:str
                ):

        self.parameterSpace = ParameterSpace(parameterDefs)