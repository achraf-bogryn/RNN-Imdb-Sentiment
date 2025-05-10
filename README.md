# 📚 Sentiment Analysis on IMDB Reviews using RNN

This project applies Recurrent Neural Networks (RNN) to perform sentiment analysis on movie reviews from the IMDB dataset. The model classifies reviews as either positive or negative based on their textual content.

## 🚀 Project Overview

Sentiment analysis is a key application in Natural Language Processing (NLP) where the goal is to determine the sentiment expressed in a piece of text. In this project, we:

- Preprocessed IMDB review data
- Built and trained an RNN model using TensorFlow/Keras
- Evaluated model performance
- Built a simple Streamlit app to make predictions on user input

## 🛠️ Tools & Technologies Used

- **Python** – Core programming language
- **Pandas & NumPy** – Data manipulation and numerical operations
- **Scikit-learn** – Data preprocessing and evaluation metrics
- **TensorFlow / Keras** – Building and training the RNN model
- **Streamlit** – Creating an interactive web interface for prediction
- **Matplotlib / Seaborn** *(optional)* – For data visualization (if used)

## 🧠 Model Architecture

- Embedding Layer
- SimpleRNN / LSTM / GRU (based on what you used)
- Dense Output Layer with Sigmoid Activation

## 📊 Dataset

- **Source**: IMDB dataset from Keras datasets (`tensorflow.keras.datasets.imdb`)
- **Content**: 25,000 labeled movie reviews (train) and 25,000 for testing

## 📈 Performance

- Accuracy: *[insert value here]*
- Loss: *[insert value here]*
- Evaluation metrics: Accuracy, Precision, Recall, F1-score

## 🌐 Streamlit App

The app allows users to input their own movie reviews and receive sentiment predictions in real-time.

### How to Run:

```bash
pip install -r requirements.txt
streamlit run app.py
