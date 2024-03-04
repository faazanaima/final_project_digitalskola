import os
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from preparation import preprocess
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def run_ml_app():
    # Specify the directory path

    st.title('Prediksi Perilaku CyberBullying')

    # Load the CountVectorizer used during training
    with open('count_vectorizer.pkl', 'rb') as file:
        count_vectorizer = pickle.load(file)

    # Input teks
    text = st.text_input("Masukkan teks bullying")

    # Create a DataFrame with the new text
    new_df = pd.DataFrame({'text': [text]})

    # Apply preprocessing to the 'text' column
    new_df['new_text_clean'] = new_df['text'].apply(preprocess)

    # Transform the new text using the fitted CountVectorizer
    X_cv_new = count_vectorizer.transform(new_df['new_text_clean'])

    # Load the TfidfTransformer used during training
    with open('tfidf_transformer.pkl', 'rb') as file:
        tfidf_transformer = pickle.load(file)

    # Transform the new text using the fitted TfidfTransformer
    X_tf_new = tfidf_transformer.transform(X_cv_new)

    # Load the pre-trained model
    with open('trained_model.sav', 'rb') as file:
        loaded_model = pickle.load(file)

    # Tombol untuk memprediksi perilaku cyberbullying
    if st.button('Hasil Prediksi'):
        # Lakukan prediksi dengan model RandomForestClassifier
        try:
            # Make predictions
            prediction = loaded_model.predict(X_tf_new)

            # Identifying labels based on prediction results
            if prediction == 0:
                bullying_detection = "Bullying Age"
            elif prediction == 1:
                bullying_detection = "Bullying Ethnicity"
            elif prediction == 2:
                bullying_detection = "Bullying Gender"
            elif prediction == 3:
                bullying_detection = "Not Cyberbullying"
            else:
                bullying_detection = "Bullying Religion"

            # Displaying prediction results
            st.text(f'Hasil Prediksi: {bullying_detection}')
        except Exception as e:
            st.error(f"Error during prediction: {e}")


# run_ml_app()
