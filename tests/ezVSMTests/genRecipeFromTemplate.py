#!python3

import numpy as np
from magLabUtilities.signalutilities.signals import SignalThread
from magLabUtilities.vsmutilities.ezVSM import insertSignalIntoQSRecipe

if __name__=='__main__':
    hAppThread = SignalThread(np.linspace(0.0, 1000.0, 11))
    recipeTemplateFP = './tests/ezVSMTests/recipeTemplate.VHC'
    recipeFP = './tests/ezVSMTests/recipe.VHC'

    insertSignalIntoQSRecipe(recipeTemplateFP, recipeFP, hAppThread)