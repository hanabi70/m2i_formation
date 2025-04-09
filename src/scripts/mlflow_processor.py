import os
import mlflow




class MLFlowProcessor:
    def __init__(self) -> None:
        self.model_name = os.getenv('MODEL_NAME','iris_model')
        self.model = None

    def predict_one(self,features:dict):
        if self.model is None:
            self.model = self.load_model()
        pred = self.model.predict(features)
        return pred[0]


    def load_model(self,model_uri:str|None = None):
        if model_uri is None:
            model_uri = self.model_name
        self.model = mlflow.pyfunc.load_model(model_uri)
        return self.model
    
            
            
    
