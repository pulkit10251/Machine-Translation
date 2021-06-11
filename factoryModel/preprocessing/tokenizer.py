from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

class TrainMaker:
    def __init__(self, MAX_NUM_WORDS):
        self.MAX_NUM_WORDS = MAX_NUM_WORDS
    
    def tokenMaker(self, text):
        tokenizer = Tokenizer(num_words = self.MAX_NUM_WORDS)
        tokenizer.fit_on_texts(text)
        return tokenizer
    
    
    def sequenceMaker(self, tokenizer, text):
        seq = tokenizer.texts_to_sequences(text)
        stdlen = self.maxLenSeq(seq)
        seq = pad_sequences(seq, maxlen = stdlen, padding = 'post')
        
        return seq, stdlen
        
    def maxLenSeq(self, lines):
        max_len =max(len(a) for a in lines)
        return max_len
    
    def preprocess(self, docArray):
        
        tokenizer_input = self.tokenMaker(docArray[:, 0])
        
        input_vocab_size= len(tokenizer_input.word_index) + 1
        
        tokenizer_target = self.tokenMaker(docArray[:, 2].tolist() + docArray[:, 3].tolist())
        
        target_vocab_size = len(tokenizer_target.word_index) + 1
        
        input_sequences, max_len_input = self.sequenceMaker(tokenizer_input, docArray[:,0])
        target_sequences_input,_ = self.sequenceMaker(tokenizer_target, docArray[:, 3])
        target_sequences, max_len_target = self.sequenceMaker(tokenizer_target, docArray[:,2])
        
        target_sequences_OneHot = to_categorical(target_sequences)
        
        return tokenizer_input,input_vocab_size,tokenizer_target, target_vocab_size,docArray, input_sequences, target_sequences, target_sequences_OneHot,target_sequences_input,max_len_input,max_len_target



    