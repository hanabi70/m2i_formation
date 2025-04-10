# Handle to the workspace
from azure.ai.ml import MLClient

# Authentication package
from azure.identity import DefaultAzureCredential

from ..core.config import config
from azure.ai.ml import command
from azure.ai.ml import Input

class AzureMLProcessor:
    def __init__(self) -> None:
        credential = DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=credential,
            subscription_id=config.AZUREML_SUBSCRIPTION_ID,
            resource_group_name=config.AZUREML_RG_NAME,
            workspace_name=config.AZUREML_WORKSPACE_NAME,
        )
        self.job = self._configure_job()
    
    def _get_path(self) -> str:
        data = self.ml_client.data.get("iris_csv", version="1")
        return data.path # type: ignore

    
    def _configure_job(self):
        job = command(
            code="./src/scripts",
            command="python azureml_trainer.py --n_estimators 100 --data ${{inputs.data}}",
            inputs={
                "data": Input(type="uri_file", path=self._get_path()),
            },
            environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest",
            compute="mlops-v2",
        )
        return job
    
    def submit_job(self):
        self.ml_client.jobs.create_or_update(self.job)


