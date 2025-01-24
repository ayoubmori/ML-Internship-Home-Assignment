import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
from pathlib import Path
from data_ml_assignment.constants import MODELS_PATH, REPORTS_PATH, LABELS_MAP

class TrainingPipeline:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model = XGBClassifier(
            n_estimators=300,  # Increased for better performance
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )

    def train(self, model_name="xgboost_model"):
        self.model.fit(self.x_train, self.y_train)
        joblib.dump(self.model, MODELS_PATH / f"{model_name}.joblib")
        
    def evaluate(self):
        preds = self.model.predict(self.x_test)
        return accuracy_score(self.y_test, preds), f1_score(self.y_test, preds, average="weighted")


    def get_model_performance(self) -> tuple:
        """
        Evaluate the model and return accuracy and F1 score.

        Returns:
            tuple: Accuracy and F1 score.
        """
        predictions = self.model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions, average="weighted")
        return accuracy, f1

    def render_confusion_matrix(self, plot_name: str = "cm_plot"):
        """
        Render and save the confusion matrix.

        Args:
            plot_name (str): Name of the confusion matrix plot file.
        """
        predictions = self.model.predict(self.x_test)
        cm = confusion_matrix(self.y_test, predictions)
        plt.rcParams["figure.figsize"] = (14, 10)

        self.plot_confusion_matrix(cm, classes=list(LABELS_MAP.values()), title="XGBoost")

        # Ensure the REPORTS_PATH directory exists
        REPORTS_PATH.mkdir(parents=True, exist_ok=True)
        plot_path = REPORTS_PATH / f"{plot_name}.png"
        plt.savefig(plot_path, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_confusion_matrix(cm, classes, title):
        """
        Helper function to plot the confusion matrix.

        Args:
            cm (numpy.ndarray): Confusion matrix.
            classes (list): List of class labels.
            title (str): Title of the plot.
        """
        import itertools
        import numpy as np

        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")