from src.application import AppManager
from src.scripts import MLFlowProcessor
from src.core.config import config

def main():

    mlflow_processor = MLFlowProcessor()
    app_manager = AppManager(mlflow_processor=mlflow_processor)
    app_manager.run(config.API_HOST, config.API_PORT)
    
if __name__ == "__main__":
    main()