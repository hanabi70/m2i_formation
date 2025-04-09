from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import os
import pandas as pd

import mlflow
from .utils import sklearn_to_frame
import argparse



class MLFlowProcessor:
    def __init__(self,model_params:dict) -> None:
        mlflow_host = os.getenv('MLFLOW_HOST','http://localhost')
        mlflow_port = os.getenv('MLFLOW_PORT','8080')
        self.mlflow_uri = f"{mlflow_host}:{mlflow_port}"
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.sklearn.autolog()
        self.iris_df = self.load_dataset()
        self.model_params = model_params
        self.model = RandomForestClassifier(**model_params) # type: ignore
        self.y = self.iris_df['target']
        self.X = self.iris_df.drop(columns=['target'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42,stratify=self.iris_df["target"])
        self.train_data = sklearn_to_frame(self.X_train,self.y_train)
        self.test_data = sklearn_to_frame(self.X_test,self.y_test)
        self.model_name = os.getenv('MODEL_NAME','iris_model')
        
    def load_dataset(self):
        # Load the iris dataset
        root_path = Path(__file__).parent.parent
        data_path = root_path.joinpath("./data/v0_iris.csv")
        iris_df = pd.read_csv(data_path)
        return iris_df
    
    def train(self):
        # Train the model
        with mlflow.start_run():
            self.model.fit(self.X_train, self.y_train)


    def predict_one(self,features:dict):
        pred = self.model.predict(features) # type: ignore
        return pred[0]

    def predict(self):
        # Make predictions on the test set
        y_pred = self.model.predict(self.X_test)
        return y_pred
    
    def compute_scores(self,y_pred):
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        dict_scores = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        return dict_scores
    
    def save_model(self):
        signature = infer_signature(self.X_train, self.model.predict(self.X_train))
        mlflow.sklearn.save_model(self.model, self.model_name, signature=signature,input_example=self.X_train.iloc[0:1])


    def load_model(self,model_uri:str|None = None):
        if model_uri is None:
            model_uri = self.model_name
        self.model = mlflow.pyfunc.load_model(model_uri)
        return self.model
    
            


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators',type=int,default=100)
    args = parser.parse_args()

    mlflow_processor = MLFlowProcessor(model_params=args.__dict__)
    print(f"mlflow uri:{mlflow_processor.mlflow_uri}")
    print("training model...")
    mlflow_processor.train()
    y_pred = mlflow_processor.predict()
    mlflow_processor.save_model()
    # metrics = mlflow_processor.compute_scores(y_pred=y_pred)
    
    model = mlflow.pyfunc.load_model("iris_model")
    print(model)
            
    
