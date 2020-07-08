#!python3

if __name__=='__main__':
    fp = './tests/workflowTests/datafiles/testLoops.xlsx'
    tuneHistoryFP = './tests/workflowTests/datafiles/tuneHistory00.txt'

    parameterDefs = {
                        'hCoercive': {'initialValue':605.0,    'limits':[0.0,10000.0]},
                        'xInit':     {'initialValue':61.25,    'limits':[60.0,90.0]},
                        'mSat':      {'initialValue':1.65e6,   'limits':[1.0e6,2.0e6]},
                        'hCoop':     {'initialValue':1190.0,   'limits':[100.0,10000.0]},
                        'hAnh':      {'initialValue':5200.0,   'limits':[100.0,10000.0]},
                        'xcPow':     {'initialValue':2.0,      'limits':[0.0,10.0]}
                    }

    gradientDescentConfig = {
                                'hCoercive': {'localNeighborSteps':[-15.0, 0.0, 15.0]},
                                'xInit':     {'localNeighborSteps':[-0.25, 0.0, 0.25]},
                                'mSat':      {'localNeighborSteps':[0.001e6, 0.0, 0.001e6]},
                                'hCoop':     {'localNeighborSteps':[-15.0, 0.0, 15.0]},
                                'hAnh':      {'localNeighborSteps':[-15.0, 0.0, 15.0]},
                                'xcPow':     {'localNeighborSteps':[-0.1, 0.0, 0.1]}
                            }
