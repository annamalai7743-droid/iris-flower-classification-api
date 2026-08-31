from pydantic import BaseModel, Field

# Input Validation Schema
class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=10, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, lt=10, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, lt=10, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, lt=10, description="Petal width in cm")

# Output Validation Schema (NEW in Task 8)
class PredictionOutput(BaseModel):
    request_id: str
    prediction: int
    confidence: float
    model_version: str = "v1.0"
    status: str = "success"