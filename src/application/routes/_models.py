from pydantic import BaseModel, Field
from typing import Optional
class PredictionInput(BaseModel):
    sepal_length: float = Field(..., description="Length of the sepal in cm")
    sepal_width: float = Field(..., description="Width of the sepal in cm")
    petal_length: float = Field(..., description="Length of the petal in cm")
    petal_width: float = Field(..., description="Width of the petal in cm")
    model_uri: Optional[str] = Field(default=None, description="URI of the model to be used for prediction")