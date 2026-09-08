from pydantic import BaseModel, ConfigDict, Field
from typing import List, Dict, Optional

class IrisInput(BaseModel):
    model_config = ConfigDict(extra="forbid")  # Extra fields-ஐ Reject செய்யும்

    sepal_length: float = Field(..., gt=0, lt=20)
    sepal_width: float = Field(..., gt=0, lt=20)
    petal_length: float = Field(..., gt=0, lt=20)
    petal_width: float = Field(..., gt=0, lt=20)

class IrisBatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
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