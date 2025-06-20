# Libraries
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import joblib


# Preprocessing function to clean the crop symptom
# Define custom stopwords for African languages
custom_stopwords = {
    'en': set(stopwords.words('english')),
    'yo': {'ní', 'rẹ̀', 'mi', 'àti', 'sí', 'pẹ̀lú'},
    'ha': {'da', 'ne', 'na', 'ka', 'zuwa'},
    'ig': {'na', 'nke', 'bụ', 'ga', 'n’'},
    'pidgin': {'na', 'dey', 'your', 'you', 'di'}
}


def preprocess_text(text, language='en'):
    # convert the text to lowercase
    text = text.lower()

    # remove punctuation and digits
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)

    # tokenize
    tokens = word_tokenize(text)

    # remove language specific stopword
    stopwords = custom_stopwords.get(language, set())
    tokens = [word for word in tokens if word not in stopwords]

    # join back into string
    return " ".join(tokens)


# read data
crop_df = pd.read_csv("data/crop_disease_symptoms.csv")
print(crop_df.head())

# Applying the preprocessing function
crop_df['Clean Text'] = crop_df.apply(
    lambda row: preprocess_text(row['Symptom Text'], row['Language']),
    axis=1
)

# Splitting the dataset into X features and y (target)
X = crop_df[['Crop', 'Clean Text', 'Language']]
y = crop_df['Disease Label']

# Splitting the X features and y target into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encoding pipeline
preprocessor = ColumnTransformer(transformers=[
    ('text', TfidfVectorizer(), 'Clean Text'),
    ('lang_crop', OneHotEncoder(), ['Crop', 'Language']),
])

# Model pipeline
pipeline = Pipeline(steps=[
 ('preprocessing', preprocessor),
 ('classifier', MultinomialNB() )
 ])

# Fitting the estimator
pipeline.fit(X_train, y_train)

# predict
y_pred = pipeline.predict(X_test)

# Evaluation metrics
print(accuracy_score(y_pred, y_test))

# Save the preprocessor
joblib.dump(preprocessor, filename="model/preprocessor.pkl")

# Save the label encoder
# joblib.dump(le, filename="model/label_encoder.pkl")

# save trained model
joblib.dump(pipeline, filename="model/pipeline.pkl")


