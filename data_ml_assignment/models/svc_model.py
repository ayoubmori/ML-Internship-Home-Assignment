from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from data_ml_assignment.models.base_model import BaseModel


class SVCModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(model=Pipeline([("svc", SVC(**kwargs))]))
