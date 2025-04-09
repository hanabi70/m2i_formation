from src.application import AppManager
from src.scripts import MLFlowProcessor
from dotenv import load_dotenv

load_dotenv(".env")

def main():

    mlflow_processor = MLFlowProcessor()
    app_manager = AppManager(mlflow_processor=mlflow_processor)
    app_manager.run()
    
if __name__ == "__main__":
    main()