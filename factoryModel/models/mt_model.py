from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, RepeatVector, Concatenate, Dot
from tensorflow.keras.layers import Input, Embedding, Lambda
import keras.backend as K



class ModelBuilding:    
    
    def __init__(self):
        pass
    
    def EncDecbuild(self,num_words, LATENT_DIM ,EMBEDDING_DIM, embedding_matrix, max_len_input, max_len_target, num_words_output):
        
        embedding_layer=Embedding(num_words,EMBEDDING_DIM,weights=[embedding_matrix],input_length=max_len_input)
        
        global encoder_inputs_placeholder,decoder_embedding,encoder_outputs,initial_s,initial_c, context_last_word_concat_layer, decoder_dense,decoder_lstm
        encoder_inputs_placeholder = Input(shape=(max_len_input,))
        x = embedding_layer(encoder_inputs_placeholder)
        encoder = Bidirectional(LSTM(
          LATENT_DIM,
          return_sequences=True,
          dropout=0.5 # dropout not available on gpu
        ))
        encoder_outputs = encoder(x)
        decoder_inputs_placeholder = Input(shape=(max_len_target,))        
        decoder_embedding = Embedding(num_words_output, EMBEDDING_DIM)
        decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)
        
        global attn_concat_layer, attn_repeat_layer, attn_dense1, attn_dense2, attn_dot
        attn_repeat_layer = RepeatVector(max_len_input)
        attn_concat_layer = Concatenate(axis=-1)
        attn_dense1 = Dense(10, activation='tanh')
        attn_dense2 = Dense(1, activation=self.softmax_over_time)
        attn_dot = Dot(axes=1)
        
        decoder_lstm = LSTM(LATENT_DIM, return_state=True)
        decoder_dense = Dense(num_words_output, activation='softmax')
        
        initial_s = Input(shape=(LATENT_DIM,), name='s0')
        initial_c = Input(shape=(LATENT_DIM,), name='c0')
        context_last_word_concat_layer = Concatenate(axis=2)
        
        
        
        s = initial_s
        c = initial_c
        
        # collect outputs in a list at first
        outputs = []

        for t in range(max_len_target):
            
            context = self.one_step_attention(encoder_outputs, s)
        
            selector = Lambda(lambda x: x[:, t:t+1]) 
            
            xt = selector(decoder_inputs_x)
            decoder_lstm_input = context_last_word_concat_layer([context, xt])
            o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[s, c])
            decoder_outputs = decoder_dense(o)
        
            outputs.append(decoder_outputs)
        
        stacker = Lambda(self.stack_and_transpose)
        outputs = stacker(outputs)
        
        # create the model
        model = Model(
          inputs=[
            encoder_inputs_placeholder,
            decoder_inputs_placeholder,
            initial_s, 
            initial_c,
          ],
          outputs=outputs
        )
        
        
        
        
        return model

            
        
        
    def stack_and_transpose(self,x):
        x = K.stack(x) 
        x = K.permute_dimensions(x, pattern=(1, 0, 2)) 
        return x
    
    def one_step_attention(self, h, st_1):
    
        st_1 = attn_repeat_layer(st_1)
    
        x = attn_concat_layer([h, st_1])
    
        x = attn_dense1(x)
        alphas = attn_dense2(x)
    
        context = attn_dot([alphas, h])
    
        return context
    
    def softmax_over_time(self, x):
        assert(K.ndim(x) > 2)
        e = K.exp(x - K.max(x,axis=1,keepdims=True))
        s = K.sum(e,axis=1,keepdims=True)
        return e/s
    
    
    def encoderDecoderModel(self, max_len_input, LATENT_DIM):
        encoder_model = Model(encoder_inputs_placeholder, encoder_outputs)

        encoder_outputs_as_input = Input(shape=(max_len_input, LATENT_DIM * 2,))
        decoder_inputs_single = Input(shape=(1,))
        decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)
        
        
        context = self.one_step_attention(encoder_outputs_as_input, initial_s)
        
        decoder_lstm_input = context_last_word_concat_layer([context, decoder_inputs_single_x])
        
        
        o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[initial_s, initial_c])
        decoder_outputs = decoder_dense(o)
        
        
        decoder_model = Model(
          inputs=[
            decoder_inputs_single,
            encoder_outputs_as_input,
            initial_s, 
            initial_c
          ],
          outputs=[decoder_outputs, s, c]
        )
        
        
        return [encoder_model, decoder_model]
            
    
    
    