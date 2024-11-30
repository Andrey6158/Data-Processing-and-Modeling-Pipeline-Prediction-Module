import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
import pickle


class TrainingHyperparameterSelection:
    """
    Class for:
    - training model and hyperparameter selection.
    """
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, model, param_grid=None, log_file='output_catboost.txt'):     
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.model = model
        self.log_file = log_file
              
        if param_grid is None:
            self.param_grid = {
                'iterations': [500, 1000],
                'depth': [6, 8],
                'learning_rate': [ 0.1, 0.2]
            }
        else:
            self.param_grid = param_grid


    def log(self, message):
        with open(self.log_file, 'a') as file:
            file.write(message + '\n')   


    def show_metrics(self, dataframes, model):
        for name, (X, y) in dataframes.items():
            pred = model.predict(X)
            mae = mean_absolute_error(y, pred)
            r2 = r2_score(y, pred)
            self.log(f"Mean Absolute Error {name}: {mae}")
            self.log(f"R-squared {name}: {r2}")
    
    
    def training_hyperparameter_selection(self):
             
        dataframes = {
            "train": (self.X_train, self.y_train),
            "val": (self.X_val, self.y_val),
            "test": (self.X_test, self.y_test)
        }
                        
        self.log("Processing training model and hyperparameter selection...")
        self.log('')

        self.model.fit(self.X_train, self.y_train, cat_features=self.X_train.select_dtypes(include=['object']).columns.tolist())
        self.log('starting Metrics:')         
        self.show_metrics(dataframes, model=self.model)
       
        # grid_search = GridSearchCV(model, self.param_grid, cv=3, scoring='r2')
        # grid_search.fit(self.X_val, self.y_val, cat_features=self.categorical_features)
        # best_params = grid_search.best_params_
        # self.log(f"Best Parameters: {best_params}\n")
                
        # best_model = grid_search.best_estimator_
        # best_model.fit(self.X_train, self.y_train, cat_features=self.categorical_features)
        # self.log('finishing Metrics:') 
        # self.show_metrics(dataframes, model=best_model)
        
        self.log('')
        self.log("Processing of saving the model...")
        with open('model.pkl', 'wb') as file:
            pickle.dump(self.model, file)   
        self.log("path: model.pkl\n")    
                        
        self.log("Processing of saving the features...")
        features = self.X_train.columns
        df_features = pd.DataFrame(features, columns=['features'])
        df_features.to_csv('features_model.csv', index=False)
        self.log("path: features_model.csv\n")