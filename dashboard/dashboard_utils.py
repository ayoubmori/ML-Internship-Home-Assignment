import requests
from data_ml_assignment.constants import SAMPLES_PATH

def getSampleText(sample):
    sample_file = "_".join(sample.upper().split()) + ".txt"
    with open(SAMPLES_PATH / sample_file, encoding="utf-8") as file:
        return file.read()

def SampleRequest(sample_text):
    return requests.post(
        "http://localhost:9000/api/inference",  # Ensure the port matches the API server
        json={"text": sample_text}
    )

from data_ml_assignment.processing.processing_data import ProcessingData
from data_ml_assignment.feature_engineering.feature_engineering_data import FeatureEngineeringData
from data_ml_assignment.training.train_pipeline import TrainingPipeline

def TrainingPipelineStats(serialize: bool, name: str):
    """
    Train the model, evaluate its performance, and return accuracy and F1 score.

    Args:
        serialize (bool): Whether to save the trained model.
        name (str): Name of the model file.

    Returns:
        tuple: Accuracy and F1 score.
    """
    # Step 1: Process data
    print("Processing data...")
    processor = ProcessingData()
    processor.process()

    # Step 2: Feature engineering
    print("Performing feature engineering...")
    fe = FeatureEngineeringData()
    X, y = fe.transform()

    # Step 3: Train model
    print("Training model...")
    tp = TrainingPipeline(X, y)
    tp.train(serialize=serialize, model_name=name)

    # Step 4: Evaluate model
    accuracy, f1 = tp.get_model_performance()
    print(f"ACCURACY = {accuracy}, F1 SCORE = {f1}")

    # Step 5: Render confusion matrix
    print("Rendering confusion matrix...")
    tp.render_confusion_matrix()

    return accuracy, f1

