import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load IMDB word index and reverse index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the trained model
model = load_model('D:\RNN-Imdb-Sentiment\SimpleRnn\imdb_Lstm_rnn_model.h5')

# Decode a review (for debugging or display)
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Preprocess the user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Streamlit App UI
st.set_page_config(page_title="IMDB Sentiment Classifier", layout="centered")
st.markdown(
    """
    <style>
    .title {
        font-size: 40px;
        color: #FF4B4B;
        font-weight: bold;
        text-align: center;
    }
    .subtext {
        text-align: center;
        font-size: 18px;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
    }
    .positive {
        background-color: #e6f9ec;
        color: #28a745;
        border: 2px solid #28a745;
    }
    .negative {
        background-color: #ffe6e6;
        color: #dc3545;
        border: 2px solid #dc3545;
    }
    </style>
    """, unsafe_allow_html=True
)

# App title and description
st.markdown('<div class="title">🎬 IMDB Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<p class="subtext">Enter a movie review to classify it as Positive or Negative.</p>', unsafe_allow_html=True)

# Text input
user_input = st.text_area('📝 Write your review here:', height=150)

# On classify
if st.button('🔍 Classify Sentiment'):
    if not user_input.strip():
        st.error("Please enter a valid movie review.")
    else:
        progress_bar = st.progress(0)
        st.write("🧠 Analyzing sentiment...")
        preprocessed_input = preprocess_text(user_input)
        progress_bar.progress(50)

        try:
            prediction = model.predict(preprocessed_input)
            sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
            prediction_score = prediction[0][0]
            progress_bar.progress(100)

            # Styled result
            if sentiment == 'Positive':
                st.markdown(
                    f'<div class="result-box positive">✅ Sentiment: Positive<br>Confidence: {prediction_score:.2f}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="result-box negative">❌ Sentiment: Negative<br>Confidence: {1 - prediction_score:.2f}</div>',
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# Optional footer
st.markdown("---")
st.markdown("📍 *Built by Achraf Bogryn | #NLP #DeepLearning #Streamlit*")
