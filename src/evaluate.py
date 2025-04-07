from loader import Loader
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
import os
import pandas as pd
class Evaluator:
    def __init__(self,loader:Loader):
        self.model = loader.load_model('model.pkl')
        self.training_data = loader.load_data('training_data.csv')
        self.testing_data = loader.load_data('testing_data.csv')
        self.X_test = self.testing_data.drop(columns=['target'])
        self.y_test = self.testing_data['target']
    
    def compute_scores(self,y_pred):
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        dict_scores = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        print(dict_scores)
        return pd.DataFrame(dict_scores, index=[0])
    def get_prediction(self):
        # Make predictions on the test set
        y_pred = self.model.predict(self.X_test)
        return y_pred

        
        

if __name__ == "__main__":
    loader = Loader()
    version = os.getenv('VERSION', '0')
    evaluator = Evaluator(loader=loader)
    y_pred = evaluator.get_prediction()
    scores = evaluator.compute_scores(y_pred=y_pred)
    loader.save_data(scores, 'scores.csv',version=version)