import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load
from pathlib import Path
from data_ml_assignment.constants import PROCESSED_DATASET_PATH, TFIDF_VECTORIZER_PATH

class FeatureEngineeringData:
    def __init__(self):
        """
        Initialize the FeatureEngineeringData class.
        """
        self.df = pd.read_csv(PROCESSED_DATASET_PATH)
        self.tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

    def fit(self):
        """
        Fit the TF-IDF vectorizer on the training data.
        """
        self.tfidf.fit(self.df["cleaned_resume"])
        # Save the fitted vectorizer for later use
        dump(self.tfidf, TFIDF_VECTORIZER_PATH)

    def transform(self):
        """
        Transform the cleaned resume text into TF-IDF features.

        Returns:
            X (scipy.sparse.csr_matrix): Transformed feature matrix.
            y (numpy.ndarray): Target labels.
        """
        # Ensure the input data is not empty
        if self.df["cleaned_resume"].empty:
            raise ValueError("The 'cleaned_resume' column is empty.")

        # Transform the text data into TF-IDF features
        X = self.tfidf.transform(self.df["cleaned_resume"])
        y = self.df["label"].values
        return X, y

    def load_vectorizer(self):
        """
        Load a pre-fitted TF-IDF vectorizer from disk.
        """
        if not Path(TFIDF_VECTORIZER_PATH).exists():
            raise FileNotFoundError(f"TF-IDF vectorizer not found at {TFIDF_VECTORIZER_PATH}.")
        self.tfidf = load(TFIDF_VECTORIZER_PATH)