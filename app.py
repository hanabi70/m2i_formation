from src.application import AppManager
from src.scripts import MLFlowProcessor


mlflow_processor = MLFlowProcessor()
app_manager = AppManager(mlflow_processor=mlflow_processor)
app = app_manager.app
    
