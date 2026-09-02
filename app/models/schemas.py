from pydantic import BaseModel
from typing import List, Dict, Optional

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisBatchInput(BaseModel):
    inputs: List[IrisInput]

class PredictionOutput(BaseModel):
    request_id: Optional[str] = None
    prediction: int
    predicted_class_name: Optional[str] = None
    confidence: float
    model_version: str
    status: Optional[str] = None

class IrisBatchOutput(BaseModel):
    predictions: List[PredictionOutput]
    batch_size: int
    status: str


class PredictionOutputV2(BaseModel):
    request_id: str
    prediction: int
    predicted_class_name: str
    class_probabilities: Dict[str, float]
    model_version: str
    status: str

class ModelInfoOutput(BaseModel):
    model_version: str
    model_type: str
    features: List[str]
    target_classes: List[str]