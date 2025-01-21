import requests
from data_ml_assignment.constants import SAMPLES_PATH
from data_ml_assignment.training.train_pipeline import TrainingPipeline


def getSampleText(sample):
    sample_file = "_".join(sample.upper().split()) + ".txt"
    with open(SAMPLES_PATH / sample_file, encoding="utf-8") as file:
        return file.read()


def SampleRequest(sample_text):
    return requests.post(
        "http://localhost:9000/api/inference", json={"text": sample_text}
    )


def TrainingPipelineStats(serialize, name):
    tp = TrainingPipeline()
    tp.train(serialize=serialize, model_name=name)
    tp.render_confusion_matrix()
    accuracy, f1 = tp.get_model_perfomance()
    return accuracy, f1
