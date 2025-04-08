import os
import mlflow
import mlflow.projects
from dotenv import load_dotenv
load_dotenv()

def run():
    uri = os.getenv('GIT_URL',"")
    print(uri)
    version = os.getenv('GIT_BRANCH',"")
    env_manager = "local"
    parameters = {
        "n_estimators":100
        }
    mlflow.projects.run(
        uri=uri,
        version=version,
        parameters=parameters,
        env_manager=env_manager
    )

if __name__ == "__main__":
    run()
