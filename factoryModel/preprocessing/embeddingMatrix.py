# create embedding matrix
import numpy as np

class Embedding:
    def __int__(self):
        pass
    
    def createWord2Vec(self, path):
        word2vec = {}

        with open(path,encoding='utf8') as f:
            for line in f:
                values=line.split()
                word=values[0]
                vector=np.asarray((values[1:]),dtype='float32')
                word2vec[word]=vector
                
        
        return word2vec
    
    def createEmbeddingMatrix(self, path ,MAX_NUM_WORDS, word2idx_inputs, EMBEDDING_DIM):
        word2vec =self.createWord2Vec(path)
        num_words = min(MAX_NUM_WORDS,len(word2idx_inputs)+1)
        embedding_matrix = np.zeros((num_words,EMBEDDING_DIM))
        for word, i in word2idx_inputs.items():
            if i < MAX_NUM_WORDS:
                embedding_vector=word2vec.get(word)
                
                if embedding_vector is not None:
                    embedding_matrix[i]=embedding_vector
                    
        return embedding_matrix

