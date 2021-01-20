#!python3

from typing import Dict

def replaceTagsInTxt(oldTxtFP:str, newTxtFP:str, replacementDict:Dict[str,str]) -> None:
    # Import old txt file
    with open(oldTxtFP) as oldFile:
        oldContents = oldFile.read()
    # Scan old txt file for keys in replacementDict; replaces keys found with associated values
    for key in replacementDict.keys():
        newContents = oldContents.replace(key, replacementDict[key])
    # Export new txt file
    with open(newTxtFP, 'w+') as newFile:
        newFile.write(newContents)