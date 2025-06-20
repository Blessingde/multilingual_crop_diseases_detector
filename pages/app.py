import streamlit as st
import pandas as pd
import joblib
from treatments import disease_info


# load model
crop_model = joblib.load('./model/pipeline.pkl')
preprocessor = joblib.load('./model/preprocessor.pkl')

# language mapping
LANGUAGE_MAPPING = {
    'English': 'en',
    'Yoruba': 'yo',
    'Hausa': 'ha',
    'Igbo': 'ig',
    'Pidgin': 'pidgin'
}

# Streamlit UI
st.title("🌱 Multilingual Crop Disease Detector")

# User input

# Ask user to describe the the symptom in their language
language_quest = {
    "English": "Please describe the symptoms you observe.",
    "Yoruba": "Jowo ṣe apejuwe àwọn ààmì arun tí o rí.",
    "Igbo": "Biko kọwaa mgbaàmà ị na-ahụ.",
    "Hausa": "Don Allah bayyana alamomin cutar da kake gani.",
    "Pidgin": "Abeg, describe wetin you see for the symptoms.",
}

# Ask the user to select their language
language = st.selectbox("Select your language:",
                     ["English", "Yoruba", "Igbo", "Hausa", "Pidgin"])

# Map the user language
lang = LANGUAGE_MAPPING[language]

symptom_text = st.text_area(language_quest[language], height=150)
crop_type = st.selectbox("Select crop type:",
                        ['Tomato', 'Maize', 'Cassava', 'Rice', 'Potato', 'Beans', 'Groundnut', 'Cocoa', 'Banana'])

if lang and symptom_text and crop_type:
    if st.button("diagnose"):
        # prepare for prediction
        input_df = pd.DataFrame({
            "Clean Text": [symptom_text],
            "Crop": [crop_type],
            "Language": [lang]
        })

        # prepare for prediction
        prediction = crop_model.predict(input_df)

        # prediction translation to other language
        other = disease_info[crop_type][prediction[0]]['label'][lang]

        # Success
        st.success(other)

        # Treatment suggestion
        suggested_treatment = disease_info[crop_type][prediction[0]]['treatment'][lang]
        st.write(suggested_treatment)

