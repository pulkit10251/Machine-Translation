'''
This is the configuration file for storing all the application parameters
'''
 
import os
from os import path
 
 
# This is the base path to the Machine Translation folder
BASE_PATH = r'C:\Users\Dell\Documents\MachineTranslation'

# Define the path where data is stored
DATA_PATH = path.sep.join([BASE_PATH,'Data/spa-eng/spa.txt'])

EMBEDDING_PATH = path.sep.join([BASE_PATH,'Data/glove.6B.100d.txt'])

BATCH_SIZE = 64
EPOCHS = 50
LATENT_DIM = 256
NUM_OF_SAMPLES = 10000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

MODEL_PATH = path.sep.join([BASE_PATH, 'factoryModel/output/model.h5'])
WEIGHTS_PATH = path.sep.join([BASE_PATH, 'factoryModel/output/weights.h5'])
INPUT_VOCAB_SIZE_PATH = path.sep.join([BASE_PATH, 'factoryModel/output/input_vocab_size.pkl'])
TOKENIZER_INPUT_PATH = path.sep.join([BASE_PATH, 'factoryModel/output/tokenizer_input.pkl'])
TOKENIZER_TARGET_PATH = path.sep.join([BASE_PATH, 'factoryModel/output/tokenizer_target.pkl'])
TARGET_VOCAB_SIZE = path.sep.join([BASE_PATH, 'factoryModel/output/target_vocab_size.pkl'])
INPUT_SEQUENCES = path.sep.join([BASE_PATH, 'factoryModel/output/input_sequences.pkl'])
TARGET_SEQUENCES = path.sep.join([BASE_PATH, 'factoryModel/output/target_sequences.pkl'])
TARGET_SEQUENCES_INPUT = path.sep.join([BASE_PATH, 'factoryModel/output/target_sequences_input.pkl'])
WORD2IDX_INPUTS = path.sep.join([BASE_PATH, 'factoryModel/output/word2idx_inputs.pkl'])
WORD2IDX_TARGET = path.sep.join([BASE_PATH, 'factoryModel/output/word2idx_target.pkl'])
MAX_LEN_INPUT = path.sep.join([BASE_PATH, 'factoryModel/output/max_len_input.pkl'])
MAX_LEN_TARGET = path.sep.join([BASE_PATH, 'factoryModel/output/max_len_target.pkl'])
ENCODER_MODEL_PATH = path.sep.join([BASE_PATH, 'factoryModel/output/encoder_model.h5'])
DECODER_MODEL_PATH = path.sep.join([BASE_PATH, 'factoryModel/output/decoder_model.h5'])
EMBEDDING_MATRIX_PATH = path.sep.join([BASE_PATH, 'factoryModel/output/embeddingMatrix.pkl'])
MODEL_OBJECT_PATH = path.sep.join([BASE_PATH, 'factoryModel/output/ModelObject.pkl'])
NUM_WORDS = path.sep.join([BASE_PATH, 'factoryModel/output/num_words.pkl'])
NUM_WORDS_OUTPUT = path.sep.join([BASE_PATH, 'factoryModel/output/num_words_output.pkl'])