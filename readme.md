# Crop Disease Symptom Classifier

## Overview
A machine learning system that classifies crop diseases from symptom descriptions in multiple African languages (English, Yoruba, Hausa, Igbo, and Pidgin).

## Features
- 🌍 Multi-language support (en, yo, ha, ig, pidgin)
- ✂️ Advanced text preprocessing with custom stopwords
- 🔢 TF-IDF text vectorization + One-hot encoding
- 📊 Multinomial Naive Bayes classifier
- 💾 Save/Load trained models with joblib

## Requirements
- Python 3.7+
- Required packages:
```bash
pip install pandas scikit-learn nltk joblib