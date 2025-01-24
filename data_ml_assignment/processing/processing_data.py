import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path
from data_ml_assignment.constants import RAW_DATASET_PATH, PROCESSED_DATASET_PATH

# Ensure the necessary NLTK data is downloaded once
nltk_data_dir = Path("nltk_data")
nltk.data.path.append(str(nltk_data_dir))  # Add the directory to NLTK's path

if not (nltk_data_dir / "corpora/stopwords").exists():
    nltk.download("stopwords", download_dir=nltk_data_dir)

if not (nltk_data_dir / "corpora/wordnet").exists():
    nltk.download("wordnet", download_dir=nltk_data_dir)


class ProcessingData:
    def __init__(self):
        self.df = pd.read_csv(RAW_DATASET_PATH)

    def clean_text(self, text):
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
