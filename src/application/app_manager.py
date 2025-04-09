from fastapi import FastAPI,APIRouter
from .routes import ApiRoutes
import uvicorn
from starlette.middleware.sessions import SessionMiddleware
from ..scripts import MLFlowProcessor
from ..core.config import config

class AppManager:
    def __init__(self,
                 mlflow_processor: MLFlowProcessor) -> None:
        self.app = FastAPI(title=config.API_TITLE)
        self.router = APIRouter()
        self.mlflow_processor = mlflow_processor
        self.api_routes = ApiRoutes(self.router,mlflow_processor)
        self.app.include_router(self.router)
        self.app.add_middleware(SessionMiddleware, secret_key="your_secret_key")
    
    def run(self,host:str = "127.0.0.1",port:int = 8000) -> None:
        uvicorn.run(self.app,host=host,port=port)
        


    