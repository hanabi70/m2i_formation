from fastapi import APIRouter
from fastapi.responses import JSONResponse
from ...scripts import MLFlowProcessor
from ._models import PredictionInput

class ApiRoutes:
    def __init__(self,
                 router:APIRouter,
                 mlflow_processor:MLFlowProcessor) -> None:
        self.router = router
        self.mlflow_processor = mlflow_processor
        self.router.post("/predict", response_class=JSONResponse)(self.predict)
    
    def predict(self,prediction_input:PredictionInput):
        prediction_args = {
            'sepal length (cm)': prediction_input.sepal_length,
            'sepal width (cm)': prediction_input.sepal_width,
            'petal length (cm)': prediction_input.petal_length,
            'petal width (cm)': prediction_input.petal_width
        }
        self.mlflow_processor.load_model(model_uri=prediction_input.model_uri)
        prediction = self.mlflow_processor.predict_one(features = prediction_args)
        return JSONResponse(content={"prediction": int(prediction)})