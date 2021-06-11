from factoryModel.config import mt_config as confFile
from factoryModel.preprocessing import SentenceSplit, cleanData, TrainMaker, Embedding
from factoryModel.dataLoaders import textLoader
from factoryModel.models import ModelBuilding
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from factoryModel.utils.helperFunctions import save_clean_data
import numpy as np


FILE_PATH = confFile.DATA_PATH


ss = SentenceSplit(10000)
cd = cleanData()
tm = TrainMaker(confFile.MAX_NUM_WORDS)
em = Embedding()

tL = textLoader(preprocessors=[ss,cd,tm])

tokenizer_input,input_vocab_size,tokenizer_target, target_vocab_size,docArray, input_sequences, target_sequences, target_sequences_OneHot,target_sequences_input, max_len_input, max_len_target = tL.loadDoc(FILE_PATH)

word2idx_inputs = tokenizer_input.word_index
word2idx_target = tokenizer_target.word_index

num_words_output = len(word2idx_target) + 1

embeddingMatrix = em.createEmbeddingMatrix(confFile.EMBEDDING_PATH, confFile.MAX_NUM_WORDS, word2idx_inputs, confFile.EMBEDDING_DIM) 

num_words = min(confFile.MAX_NUM_WORDS,len(word2idx_inputs)+1)

Model = ModelBuilding()
model = Model.EncDecbuild(num_words, confFile.LATENT_DIM, confFile.EMBEDDING_DIM, embeddingMatrix, max_len_input, max_len_target, num_words_output)
model.compile(optimizer='adam',loss='categorical_crossentropy')
checkpoint = ModelCheckpoint('model.h5', monitor = 'loss', verbose = 1 , save_best_only = True, mode = min)

z = np.zeros((len(input_sequences), confFile.LATENT_DIM))
"""
h = model.fit(
  [input_sequences, target_sequences_input, z, z], target_sequences_OneHot,
  batch_size=confFile.BATCH_SIZE,
  epochs=confFile.EPOCHS,
  callbacks = [checkpoint]
)
"""

model.save_weights("weights.h5")
model.load_weights("weights.h5")

save_clean_data(embeddingMatrix, 'embeddingMatrix.pkl')
save_clean_data(num_words, 'num_words.pkl')
save_clean_data(num_words_output, 'num_words_output.pkl')
save_clean_data(tokenizer_input,'tokenizer_input.pkl')
save_clean_data(input_vocab_size,'input_vocab_size.pkl')
save_clean_data(tokenizer_target,'tokenizer_target.pkl')
save_clean_data(target_vocab_size,'target_vocab_size.pkl')
save_clean_data(input_sequences,'input_sequences.pkl')
save_clean_data(target_sequences,'target_sequences.pkl')
save_clean_data(target_sequences_input,'target_sequences_input.pkl')
save_clean_data(word2idx_inputs,'word2idx_inputs.pkl')
save_clean_data(word2idx_target,'word2idx_target.pkl')
save_clean_data(max_len_input, 'max_len_input.pkl')
save_clean_data(max_len_target, 'max_len_target.pkl')


infmodels = Model.encoderDecoderModel(max_len_input, confFile.LATENT_DIM)



