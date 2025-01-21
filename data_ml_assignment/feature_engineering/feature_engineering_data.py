import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from data_ml_assignment.constants import PROCESSED_DATASET_PATH

class FeatureEngineeringData:
    def __init__(self):
        self.df = pd.read_csv(PROCESSED_DATASET_PATH)
        self.tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

    def transform(self):
        X = self.tfidf.fit_transform(self.df["cleaned_resume"])
        y = self.df["label"]
        return X, y