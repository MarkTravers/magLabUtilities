#!python3

class Error(Exception):
    pass

class IncompatibleArgumentError(Error):
    pass

class DataShapeError(Error):
    pass