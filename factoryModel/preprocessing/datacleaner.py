
import string
from numpy import array

class cleanData:
    def __init__(self):
        # Creating the constructor for removing punctuations and lowering the text
        pass
         
    # Creating the function for removing the punctuations and converting to lowercase
    def preprocess(self,lines):
        cleanArray = list()
        for docs in lines:
            cleanDocs = list()
            for line in docs:
                line = [word.lower() for word in line]
                cleanDocs.append(''.join(line))
            cleanArray.append(cleanDocs)
        return array(cleanArray)