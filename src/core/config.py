"""Config module"""

from pydantic_settings import BaseSettings,SettingsConfigDict

class Config(BaseSettings):
    """env variables"""
    # GENERAL
    API_TITLE: str
    API_HOST: str
    API_PORT: int
    MLFLOW_HOST:str
    MLFLOW_PORT:int
    GIT_URI:str
    GIT_BRANCH:str
    VERSION:int
    MLFLOW_EXPERIMENT_NAME:str
    MODEL_NAME:str
    MODEL_URI:str
    MLFLOW_TRACKING_URI:str


    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        case_sensitive = True,
        validate_assignment = True
    )

config = Config() # type: ignore