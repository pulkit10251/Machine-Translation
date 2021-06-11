'''
This script lists down all the helper functions which are required for processing raw data
'''
 
from pickle import load
from numpy import argmax, array

from keras.preprocessing.sequence import pad_sequences
from pickle import dump
 
def save_clean_data(data,filename):
    dump(data,open(filename,'wb'))
    print('Saved: %s' % filename)
    

def load_files(filename):
    return load(open(filename,'rb'))


def cleanInput(lines):
    cleanSent = []
    cleanDocs = list()
    for docs in lines[0].split():
        line = docs[0].lower()
        cleanDocs.append(line)
    cleanSent.append(''.join(cleanDocs))
    return array(cleanSent)



def encode_sequences(tokenizer,length,lines):
    X = tokenizer.texts_to_sequences(lines)
    X = pad_sequences(X,maxlen=length,padding='post')
    return X


# Generate target sentence given source sequence
def Convertsequence(tokenizer,source):
    target = list()
    reverse_eng = tokenizer.index_word
    for i in source:
        if i == 0:
            continue
        target.append(reverse_eng[int(i)])
    return ' '.join(target)



# Function to generate predictions from source data
def generatePredictions(model,tokenizer,data):
    prediction = model.predict(data,verbose=0)    
    AllPreds = []
    for i in range(len(prediction)):
        predIndex = [argmax(prediction[i, :, :], axis=-1)][0]
        target = Convertsequence(tokenizer,predIndex)
        AllPreds.append(target)
    return AllPreds