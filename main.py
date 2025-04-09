import os
import mlflow
import mlflow.projects
from dotenv import load_dotenv
load_dotenv(".env")

def run():
    uri = os.getenv('GIT_URI',"")
    version = os.getenv('GIT_BRANCH',"")
    experiment_name = os.getenv('EXPERIMENT_NAME',"")
    env_manager = "local"
    parameters = {
        "n_estimators":100
        }
    mlflow.projects.run(
        uri=uri,
        version=version,
        parameters=parameters,
        env_manager=env_manager,
        experiment_name=experiment_name,
    )

if __name__ == "__main__":
    run()
