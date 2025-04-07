from sklearn import datasets
import pandas as pd
from loader import Loader
import os
class DataPreparation:
    def __init__(self):
        self.iris_df = self.load_dataset()
    
    def load_dataset(self):
        # Load the iris dataset
        iris = datasets.load_iris()
        iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        iris_data['target'] = iris.target
        return iris_data

    

if __name__ == "__main__":
    data_prep = DataPreparation()
    loader = Loader()
    version = os.getenv('VERSION', '0')
    loader.save_data(data_prep.iris_df, 'iris.csv',version=version)

