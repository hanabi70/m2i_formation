from src.scripts.azureml_processor import AzureMLProcessor
from dotenv import load_dotenv
load_dotenv(".env")


if __name__ == "__main__":
    azureml_processor = AzureMLProcessor()
    azureml_processor.submit_job()