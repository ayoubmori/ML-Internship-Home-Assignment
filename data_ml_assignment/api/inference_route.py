from fastapi import APIRouter, HTTPException
from data_ml_assignment.models.xgbc_model import XGBCModel
from data_ml_assignment.constants import NEW_MODEL_PATH
import joblib
from data_ml_assignment.api.schemas import Resume
from data_ml_assignment.constants import LABELS_MAP


inference_router = APIRouter()

model = XGBCModel()
model.load(NEW_MODEL_PATH)  # Load XGBoost model

@inference_router.post("/inference")
def run_inference(resume: Resume):
    try:
        # Perform prediction
        prediction = model.predict([resume.text])

        # Convert prediction to a Python native type (e.g., int)
        predicted_label = int(prediction.tolist()[0])

        return {"label": predicted_label}
    except Exception as e:
        # Return a 500 Internal Server Error with a descriptive message
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )
