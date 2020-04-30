#!python3

import numpy as np

# Calculates and formats field values to create a quasi-static FORC for EZVSM
class Forc:
    # Constructs the FORC and calculates step data
    def __init__(self, satField, reversalField, satToSatPoints, minFieldStepSize):
        print('Creating FORC for reversal = %.4f' % (reversalField))
        self.satField = satField
        self.reversalField = reversalField
        self.satToSatPoints = satToSatPoints
        self.minFieldStepSize = minFieldStepSize

        # User input checking
        if self.satToSatPoints % 2 != 1:
            raise Exception('satToSatPoints: Expected an odd integer.')
        self.satToZeroPoints = (self.satToSatPoints-1)/2
        if self.satField < 0:
            raise Exception('satField: Expected a positive saturation field.')

        # Calculates 'a' and 'b' respectively for y = (a*x)^b such that y(x) maps data points to an H-field value.
        self.fieldMappingOrder = np.log10(minFieldStepSize / satField) / np.log10(1.0 / self.satToZeroPoints)
        self.fieldMappingFactor = np.power(self.minFieldStepSize, 1.0 / self.fieldMappingOrder) / 1.0

        # Calculates the data for each H-field step at each data point
        self.stepData = None
        self.calcFieldStepData()

    def calcFieldStepData(self):
        fieldValueArray = np.empty((0,2))
        fieldIndex = self.satToZeroPoints
        # Steps through the data points in the FORC
        while (True):
            fieldValue = self.calcField(fieldIndex)
            nextFieldValue = self.calcField(fieldIndex - 1)
            fieldIndex -= 1

            # Breaks the loop when the FORC reaches the reversal field; sets the last H-field step exactly to the reversal field.
            if nextFieldValue < self.reversalField:
                fieldValueArray = np.vstack((fieldValueArray, np.array([fieldValue, self.reversalField])))
                break
            # Add the next H-field step to the step data
            else:
                fieldValueArray = np.vstack((fieldValueArray, np.array([fieldValue, nextFieldValue])))

        # Mirrors the step data for the FORC reversal leg
        self.stepData = np.vstack((fieldValueArray, np.flip(np.flip(fieldValueArray, axis=0), axis=1)))
        # Appends the H-field step size to the step data
        self.stepData = np.hstack((self.stepData, np.abs(np.subtract(self.stepData[:,0:1], self.stepData[:,1:2]))))

    # Calculates the H-field strength at a particular data point
    def calcField(self, fieldIndex):
        # Negate the field if the field index is negative
        if fieldIndex > 0:
            return(np.power(self.fieldMappingFactor * fieldIndex, self.fieldMappingOrder))
        else:
            return(-np.power(self.fieldMappingFactor * np.abs(fieldIndex), self.fieldMappingOrder))

class RecipeAssembler:
    def __init__(self, forcFilename, recipeTemplateFilename, recipeStepDataFilename, recipeOutputFilename):
        self.forcFilename = forcFilename
        self.recipeTemplateFilename = recipeTemplateFilename
        self.recipeStepDataFilename = recipeStepDataFilename
        self.recipeOutputFilename = recipeOutputFilename

        self.forcList = []
        self.forcStepData = np.empty((0,3))
        self.importForcs()
        self.vsmCommandString = ''
        self.generateVsmCommands()
        self.generateRecipeFile()

    def importForcs(self):
        print('Reading FORC list file...')
        forcFileData = np.genfromtxt(self.forcFilename, comments='#', skip_header=1)
        # Handle the case where there is only one FORC in the input file
        if len(forcFileData.shape) < 2:
            forcFileData = forcFileData.reshape((1,4))
        print('Creating FORCs...')
        # Create a FORC instance for each FORC requested in the input file
        for forcIndex in range(forcFileData.shape[0]):
            forcSaturation = forcFileData[forcIndex, 0]
            forcReversal = forcFileData[forcIndex, 1]
            satToSatPoints = forcFileData[forcIndex, 2]
            minHstep = forcFileData[forcIndex, 3]

            self.forcList.append(Forc(forcSaturation, forcReversal, satToSatPoints, minHstep))
            self.forcStepData = np.vstack((self.forcStepData, self.forcList[-1].stepData))
        
        # Dump a text file with all step data for debugging
        np.savetxt(self.recipeStepDataFilename, self.forcStepData, fmt='%.4f', delimiter='      ')

    def generateVsmCommands(self):
        # Converts step data to VSM commands
        print('Generating VSM recipe...')
        self.vsmCommandString = '@Number of sections= %d\n' % (self.forcStepData.shape[0])
        self.vsmCommandString = self.vsmCommandString + '@Section 0: Hysteresis; New Plot\n'
        self.vsmCommandString = self.vsmCommandString + '@Preparation Actions:\n'
        self.vsmCommandString = self.vsmCommandString + 'Action 0:      Set Gauss Range to 4.0000 [ ] and wait 0.0000 s ; Set Mode = Set and wait till there\n'
        self.vsmCommandString = self.vsmCommandString + '@Repeated Actions:\n'
        self.vsmCommandString = self.vsmCommandString + 'Action 0:      Set Applied Field to 0.0000 [Oe] and wait 0.0000 s ; Set Mode = Set and wait till there; Measure \n'
        self.vsmCommandString = self.vsmCommandString + '@Main Parameter = 0 : Applied Field [Oe].\n'
        self.vsmCommandString = self.vsmCommandString + '@Main Parameter Setup:\n'
        self.vsmCommandString = self.vsmCommandString + '     From: %.4f [Oe] To: %.4f [Oe] Min Stepsize/Sweeprate = %.4f [Oe] Max Stepsize/Sweeprate = %.4f [Oe]\n' % (self.forcStepData[0][0], self.forcStepData[0][1], self.forcStepData[0][2], self.forcStepData[0][2])
        self.vsmCommandString = self.vsmCommandString + '     Signal change min step =  0.00 [%] Signal change max step =  0.00 [%] ;  Wait time =    0.00 [sec] Up & Down = No\n'
        self.vsmCommandString = self.vsmCommandString + '@Measured Signal(s) = X & Y\n'
        self.vsmCommandString = self.vsmCommandString + '@Section 0 END\n'
        for forcStep in range(1, self.forcStepData.shape[0]):
            self.vsmCommandString = self.vsmCommandString + '@Section %d: Hysteresis\n' % (forcStep)
            self.vsmCommandString = self.vsmCommandString + '@Main Parameter Setup:\n'
            self.vsmCommandString = self.vsmCommandString + '     From: %.4f [Oe] To: %.4f [Oe] Min Stepsize/Sweeprate = %.4f [Oe] Max Stepsize/Sweeprate = %.4f [Oe]\n' % (self.forcStepData[forcStep][0], self.forcStepData[forcStep][1], self.forcStepData[forcStep][2], self.forcStepData[forcStep][2])
            self.vsmCommandString = self.vsmCommandString + '     Signal change min step =  0.00 [%] Signal change max step =  0.00 [%] ;  Wait time =    0.00 [sec] Up & Down = No\n'
            self.vsmCommandString = self.vsmCommandString + '@Section %d END\n' % (forcStep)

    def generateRecipeFile(self):
        # Exports the VSM commands to a recipe file
        print('Exporting VSM recipe file...')
        recipeTemplateData = ''
        with open(self.recipeTemplateFilename) as recipeTemplateFile:
            recipeTemplateData = recipeTemplateFile.read()
        recipeTemplateData = recipeTemplateData.replace('__insert_section_data_here__\n', self.vsmCommandString)
        with open(self.recipeOutputFilename, 'w+') as recipeOutputFile:
            recipeOutputFile.write(recipeTemplateData)

if __name__ == '__main__':
    print('\n#### VSM FORC Recipe Generator ####\nStarting...')
    # RecipeAssembler(forcFilename, recipeTemplateFilename, recipeStepDataFilename, recipeOutputFilename)
    RecipeAssembler('./sample_params/1kG_majorLoop.txt', './recipe_templates/Hys-a000-RT-87_AutoRange_template.VHC', './recipes/1kG_majorLoop.txt', './recipes/1kG_majorLoop.VHC')
    print('Done.')
