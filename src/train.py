# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
from loader import Loader

class Trainer:
    def __init__(self,loader:Loader):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.iris_df = loader.load_data('iris.csv')
        self.y = self.iris_df['target']
        self.X = self.iris_df.drop(columns=['target'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42,stratify=self.iris_df["target"])

    def get_training_data(self):
        return pd.concat([self.X_train, self.y_train], axis=1)
    def get_testing_data(self):
        return pd.concat([self.X_test, self.y_test], axis=1)    

    def train(self):
        # Train the model
        self.model.fit(self.X_train, self.y_train)
    def predict(self):
        # Make predictions on the test set
        y_pred = self.model.predict(self.X_test)
        return y_pred

if __name__ == "__main__":
    loader = Loader()
    trainer = Trainer(loader=loader)
    trainer.train()
    training_data = trainer.get_training_data()
    testing_data = trainer.get_testing_data()
    version = os.getenv('VERSION', '0')
    loader.save_data(training_data, 'training_data.csv',version=version)
    loader.save_data(testing_data, 'testing_data.csv',version=version)
    loader.save_model(trainer.model, 'model.pkl',version=version)
    # Save the model
