import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb # type: ignore
from tensorflow.keras.preprocessing import sequence # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import streamlit as st



# Step 1: Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('D:\RNN-Imdb-Sentiment\SimpleRnn\imdb_rnn_model.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # Ensure unknown words get an index
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)  # Padding for fixed length
    return padded_review

# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('### Enter a movie review below to classify it as positive or negative.')

# Add an image or background if needed
# st.image("https://via.placeholder.com/400x100.png?text=IMDB+Sentiment+Analysis", use_column_width=True)

# User input area
user_input = st.text_area('Enter Movie Review', height=150)

# Make the button actionable
if st.button('Classify Sentiment'):
    # Ensure the user has inputted text
    if not user_input.strip():
        st.error("Please enter a valid movie review!")
    else:
        # Display a progress bar while processing
        progress_bar = st.progress(0)
        st.write("Processing your review...")

        # Preprocess the text
        preprocessed_input = preprocess_text(user_input)
        progress_bar.progress(50)

        # Make prediction
        try:
            prediction = model.predict(preprocessed_input)
            sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
            prediction_score = prediction[0][0]

            # Update progress bar to 100%
            progress_bar.progress(100)

            # Display results
            st.write(f"### Sentiment: {sentiment}")
            # st.write(f"Prediction Score: {prediction_score:.4f}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
else:
    st.write("### Please enter a review and click 'Classify Sentiment' to see the result.")
