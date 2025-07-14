import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load trained model
model = load_model('simple.h5')

# Function to decode tokenized reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess raw text
def preprocess_text(text, max_len=500):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=max_len)
    return padded_review, encoded_review  # return both padded & encoded

# Streamlit UI
st.title("Movie Review Sentiment Classifier 🎬")

user_input = st.text_area('Enter a Movie Review:')

if st.button('Classify'):
    padded_input, encoded_input = preprocess_text(user_input)

    prediction = model.predict(padded_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    decoded_input = decode_review(encoded_input)

    st.subheader("Your Processed Review:")
    st.write(decoded_input)

    st.subheader("Prediction Result:")
    st.write(f'Sentiment: **{sentiment}**')
    st.write(f'Prediction Score: `{prediction[0][0]:.4f}`')
else:
    st.write("⬆️ Please enter a review above and click 'Classify'.")
