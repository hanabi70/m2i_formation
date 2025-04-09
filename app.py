from src.application import AppManager
from src.scripts import MLFlowProcessor
from dotenv import load_dotenv

load_dotenv(".env")

def main():
    model_params = {"n_estimators": 100}
    mlflow_processor = MLFlowProcessor(model_params=model_params)
    app_manager = AppManager(mlflow_processor=mlflow_processor)
    app_manager.run()
    
if __name__ == "__main__":
    main()