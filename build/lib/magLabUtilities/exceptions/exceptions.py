#!python3

### Exceptions ###
class Error(Exception):
    pass

class SignalTypeError(TypeError):
    pass

class SignalValueError(ValueError):
    pass

class FileIOValueError(ValueError):
    pass

class CostValueError(ValueError):
    pass

class UITypeError(TypeError):
    pass

class UIValueError(ValueError):
    pass

### Warnings ###
import warnings

class DatafileImportWarning(Warning):
    pass
