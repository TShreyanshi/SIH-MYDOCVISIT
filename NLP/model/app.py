import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, f1_score, jaccard_score
import joblib
import nltk
import streamlit as st

df = pd.read_csv('E:/SIH/NLP/processed_data.csv')
file = "E:/SIH/NLP/Doctors_name.csv"
df_doc = pd.read_csv(file)
df_doc.rename(columns={"Doctor's Name": 'DoctorName'}, inplace=True)
df_doc = df_doc.drop_duplicates()
df_doc = df_doc.reset_index(drop=True)
df_doc = df_doc.head(1640)

df['Combined'] = df['Disease'] + ' ' + df['Symptoms'] + ' ' + df['Precautions']

tfidf_vectorizer = TfidfVectorizer()
mlb = MultiLabelBinarizer()

X = tfidf_vectorizer.fit_transform(df['Combined'])
y = mlb.fit_transform(df['Specialities'].str.split(','))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=100))
classifier.fit(X_train, y_train)

df = pd.concat([df, df_doc], axis=1)

st.set_page_config(
    page_title="Doctor Speciality Prediction",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    body {
        background-color: #ADD8E6;
    }
    .stApp {
        max-width: 1800px;
        padding: 2rem;
        background-color: transparent; /* Set the background of the main content area to transparent */
    }
    .stHeader {
        color: #333;
        font-size: 24px;
        text-align: center;
    }
    .stTextInput {
        margin-top: 1rem;
        padding: 0.5rem;
        font-size: 20px;
    }
    .stSubheader {
        font-size: 20px;
        margin-top: 2rem;
    }
    .stResults {
        margin-top: 1rem;
        font-size: 18px;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title('Doctor Speciality Prediction')

user_input = st.text_input("Enter a Disease or Symptoms:")
if user_input:
    user_input = tfidf_vectorizer.transform([user_input])

    predictions = classifier.predict(user_input)

    predicted_specialities = mlb.inverse_transform(predictions)

    if predicted_specialities:
        if predicted_specialities[0]:
            predicted_speciality = predicted_specialities[0][0]
        else:
            predicted_speciality = "No Doctors have speciality related to this Disease"
    else:
        predicted_speciality = "No matching speciality found"

    st.subheader("Predicted Doctor Speciality:")
    st.write(predicted_speciality)

    def doctor(input_value):
        if 'DoctorName' in df.columns:
            doctor_names = df[df['Specialities'] == input_value]['DoctorName'].tolist()
            return doctor_names
        else:
            return []

    if predicted_speciality != "No Doctors have speciality related to this Disease":
        doctor_names = doctor(predicted_speciality)
        if doctor_names:
            st.subheader("Doctors Available for the Predicted Speciality:")
            for doctor_name in doctor_names:
                st.write(doctor_name.replace('\xa0', ' '))
        else:
            st.write("No doctors available for this speciality.")
    else:
        st.write("Doctor is not available")

