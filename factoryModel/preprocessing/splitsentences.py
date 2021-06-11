'''
Script for preprocessing of text for Machine Translation
This is the class for splitting the text into sentences
'''
 
import string
import numpy as np

class SentenceSplit:
    def __init__(self,nrecords):
        # Creating the constructor for splitting the sentences
        # nrecords is the parameter which defines how many records you want to take from the data set
        self.nrecords = nrecords
         
    # Creating the new function for splitting the text
    def preprocess(self,text):
        sen = text.strip().split('\n')
        sen = [i.split('\t') for i in sen]
        # Saving into an array
        sen = np.array(sen)
        sen = sen[:self.nrecords,:2]
        target_text = []
        target_text_input = []
        for i in range(self.nrecords):
            target_text.append(sen[i,1]+"<eos>")
            target_text_input.append("<sos>"+ sen[i,1])

        target_text = np.array(target_text)
        target_text_input = np.array(target_text_input)
        target_text = target_text.reshape(target_text.shape[0],1)
        target_text_input = target_text_input.reshape(target_text_input.shape[0], 1)
        sen = np.append(sen, target_text, axis = 1)
        sen = np.append(sen, target_text_input, axis = 1)
        
        # Return only the first two columns as the third column is metadata. Also select the number of rows required
        return sen
    
    