import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path
from data_ml_assignment.constants import RAW_DATASET_PATH, PROCESSED_DATASET_PATH

class ProcessingData:
    def __init__(self):
        self.df = pd.read_csv(RAW_DATASET_PATH)

    def clean_text(self, text):
        nltk.download("stopwords")
        nltk.download("wordnet")

        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))

        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\d+", "", text)
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
        return text

    def process(self):
        self.df["cleaned_resume"] = self.df["resume"].apply(self.clean_text)
        self.df.to_csv(PROCESSED_DATASET_PATH, index=False)