import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import pickle

class Predict:
    """
    Class:
    - Takes as input a dataset for which the target variable needs to be predicted;
    - Accepts a pre-trained model as input;
    - Accepts a set of selected features as input;
    - Performs prediction of the target variable based on these inputs;
    - Adds the predicted values, a prediction interval (±20% of the predicted value), and a label indicating whether the true target variable falls within this interval to the original dataset;
    - Returns the updated dataset with the added information.
    """
    def __init__(self, path_data, path_model, path_features,  log_file='output_catboost.txt'):     
        self.data = pd.read_csv(path_data)
        with open(path_model, 'rb') as file:
              self.model = pickle.load(file)
        self.features = pd.read_csv(path_features)
        self.log_file = log_file
    

    def log(self, message):
        with open(self.log_file, 'a') as file:
            file.write(message + '\n')   


    def predict(self):
        self.log(f'Prediction process...')
        self.data.dropna(inplace=True)
        y = self.data['price']
        self.features = self.features['features'].tolist()
        self.X = self.data[self.features]
        pred = self.model.predict(self.X).round(0)
        r2 = r2_score(y, pred)
        self.log(f"R-squared: {r2}")
        self.data['pred'] = pred
        self.data['low'] = self.data['pred'] * 0.8
        self.data['high'] = self.data['pred'] * 1.2
        conditions = [self.data['price'] < self.data['low'], self.data['price'] > self.data['high']]
        choices = ['UNDERPRICE', 'OVERPRICE']
        self.data['price_grade'] = np.select(conditions, choices, default='real_price')
        self.log('==============================================')
        self.log(f"{self.data[['price', 'pred', 'low', 'high', 'price_grade']]}")
        self.data.to_csv('dataset_output.csv', index=False)
        self.log('=============================================================================') 