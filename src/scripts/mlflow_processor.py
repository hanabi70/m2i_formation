import mlflow
from ..core.config import config

class MLFlowProcessor:
    def __init__(self) -> None:
        self.model = self.load_model(model_uri=config.MODEL_URI)

    def predict_one(self,features:dict):
        pred = self.model.predict(features)
        return pred[0]

    def load_model(self,model_uri:str):
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    
            
            
    
