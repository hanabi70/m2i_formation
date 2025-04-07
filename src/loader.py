import pathlib
import pandas as pd
import joblib
class Loader:
    def __init__(self):
        self.path = pathlib.Path(__file__).parent.parent.resolve()
       
    def save_data(self,data:pd.DataFrame, filename:str,version:str = "0"):
        save_path = f"{self.path}/data/process"
        path = f'{save_path}/v{version}_{filename}'
        data.to_csv(path, index=False)
        
    def load_data(self,filename:str,version:str = "0"):
        load_path = f"{self.path}/data/process"
        path = f'{load_path}/v{version}_{filename}'
        df = pd.read_csv(path)
        return df
    
    def save_model(self,model,filename:str,version:str = "0"):
        save_path = f"{self.path}/model"
        path = f'{save_path}/v{version}_{filename}'
        joblib.dump(model, path)
    
    def load_model(self,filename:str,version:str = "0"):
        load_path = f"{self.path}/model"
        path = f'{load_path}/v{version}_{filename}'
        model = joblib.load(path)
        return model