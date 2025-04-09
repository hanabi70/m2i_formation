from pydantic import BaseModel, Field
class PredictionInput(BaseModel):
    sepal_length: float = Field(..., description="Length of the sepal in cm")
    sepal_width: float = Field(..., description="Width of the sepal in cm")
    petal_length: float = Field(..., description="Length of the petal in cm")
    petal_width: float = Field(..., description="Width of the petal in cm")

class PredictionOutput(BaseModel):
    prediction: float = Field(..., description="Predicted class label")