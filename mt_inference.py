from factoryModel.utils.helperFunctions import load_files
from factoryModel.config import mt_config as confFile
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from factoryModel.models import ModelBuilding
import numpy as np


    
def decode_sequence(input_sequence, encoder_model, decoder_model):
    
    word2idx_input_path = confFile.WORD2IDX_INPUTS
    word2idx_output_path = confFile.WORD2IDX_TARGET

    
    word2idx_input = load_files(word2idx_input_path)
    word2idx_target = load_files(word2idx_output_path)
    idx2word_input = {v:k for k, v in word2idx_input.items()}
    idx2word_trans = {v:k for k, v in word2idx_target.items()}
    enc_out = encoder_model.predict(input_sequence)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx_target['sos']
    eos = word2idx_target['eos']
    
    s = np.zeros((1, confFile.LATENT_DIM))
    c = np.zeros((1, confFile.LATENT_DIM))


    output_sentence = []
    for _ in range(max_len_target):
        o, s, c = decoder_model.predict([target_seq, enc_out, s, c])

        idx = np.argmax(o.flatten())

        # End sentence of EOS
        if eos == idx:
            break

        word = ''
        if idx > 0:
            word = idx2word_trans[idx]
            output_sentence.append(word)


        target_seq[0, 0] = idx

    return ' '.join(output_sentence)

    


def test(string, infrence_models):
    tokenizerpath = confFile.TOKENIZER_INPUT_PATH
    max_len_input_path = confFile.MAX_LEN_INPUT
    max_len_input = load_files(max_len_input_path)
    tokenizer_input = load_files(tokenizerpath)
    string_seq=tokenizer_input.texts_to_sequences([string])
    string_seq=pad_sequences(string_seq,maxlen=max_len_input)
    translation=decode_sequence(string_seq, infrence_models[0], infrence_models[1])
    
    return translation



Model = ModelBuilding()

embedding_matrix = load_files(confFile.EMBEDDING_MATRIX_PATH)
max_len_target_path = confFile.MAX_LEN_TARGET
max_len_input_path = confFile.MAX_LEN_INPUT
model_weights_path = confFile.WEIGHTS_PATH
max_len_target = load_files(max_len_target_path)
max_len_input = load_files(max_len_input_path)
num_words = load_files(confFile.NUM_WORDS)
num_words_output = load_files(confFile.NUM_WORDS_OUTPUT)

model = Model.EncDecbuild(num_words, confFile.LATENT_DIM, confFile.EMBEDDING_DIM, embedding_matrix, max_len_input, max_len_target, num_words_output)

model.load_weights(model_weights_path)

infrence_models = Model.encoderDecoderModel(max_len_input, confFile.LATENT_DIM)

translation = test("tom is laughing", infrence_models)

print(translation)


   