import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence


model=load_model('lstm_model.h5')

with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)
    
def prediction(model,tokenizer,text,max_sequence_len):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if(len(token_list) >max_sequence_len):
        token_list=token_list[-(max_sequence_len-1):]
    padded_sequence=sequence.pad_sequences([token_list],maxlen=max_sequence_len-1)
    predicted=model.predict(padded_sequence,verbose=1)
    predicted_word_index=np.argmax(predicted,axis=1)
    for word,index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    return None

st.title('Next Word Prediction with LSTM')
input_text=st.text_input('Enter the text0')
if st.button('Predict Next Word'):
    max_len=model.input_shape[1]+1
    next_word=prediction(model,tokenizer,input_text,max_len) 
    st.write(f'Next Word:{next_word}')