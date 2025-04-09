from fastapi import APIRouter
from ...scripts import MLFlowProcessor
from ._models import PredictionInput, PredictionOutput

class ApiRoutes:
    def __init__(self,
                 router:APIRouter,
                 mlflow_processor:MLFlowProcessor) -> None:
        self.router = router
        self.mlflow_processor = mlflow_processor
        self.router.post("/predict", response_model=PredictionOutput)(self.predict)
    
    async def predict(self,prediction_input:PredictionInput):
        # prediction_args = {
        #     'sepal length (cm)': prediction_input.sepal_length,
        #     'sepal width (cm)': prediction_input.sepal_width,
        #     'petal length (cm)': prediction_input.petal_length,
        #     'petal width (cm)': prediction_input.petal_width
        # }
        prediction = self.mlflow_processor.predict_one(features = prediction_input.model_dump(mode="python"))
        return PredictionOutput(prediction=prediction)