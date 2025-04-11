from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature
import pandas as pd
import os
import mlflow
import argparse



class MLFlowTrainer:
    def __init__(self,model_params:dict) -> None:
        mlflow_host = os.getenv("MLFLOW_HOST")
        mlflow_port = os.getenv("MLFLOW_PORT")
        self.mlflow_uri = f"{mlflow_host}:{mlflow_port}"
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.sklearn.autolog()
        self.iris_df = self.load_dataset()
        self.model_params = model_params
        self.model = RandomForestClassifier(**self.model_params)
        self.y = self.iris_df['target']
        self.X = self.iris_df.drop(columns=['target'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42,stratify=self.iris_df["target"])
        self.train_data = self.sklearn_to_frame(self.X_train,self.y_train)
        self.test_data = self.sklearn_to_frame(self.X_test,self.y_test)
        self.model_name = os.getenv("MODEL_NAME","iris_model")
        
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
    
    def save_model(self):
        signature = infer_signature(self.X_train, self.model.predict(self.X_train))
        mlflow.sklearn.save_model(self.model, self.model_name, signature=signature,input_example=self.X_train.iloc[0:1])

    def sklearn_to_frame(self,X,y) -> pd.DataFrame:    
        return pd.concat([X, y], axis=1)

    
            


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators',type=int,default=100)
    args = parser.parse_args()

    mlflow_processor = MLFlowTrainer(model_params=args.__dict__)
    print(f"mlflow uri:{mlflow_processor.mlflow_uri}")
    print("training model...")
    mlflow_processor.train()
    mlflow_processor.save_model()
    # metrics = mlflow_processor.compute_scores(y_pred=y_pred)


            
    
