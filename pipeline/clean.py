import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


#columns_to_droop = ['deposit', 'cian_id', 'time_add', 'description', 'url', 'uuid', 'house']

class Clean:
    def __init__(self, path, target_name,  columns_to_droop, log_file='output_catboost.txt'): 
        """
        Class for preprocessing a dataset, including:
        - removing unnecessary features for modeling;
        - removing duplicates;
        - processing outliers of data;
        - separating features by type;
        - encoding of object features;
        - processing nan values;             
        - removing features with high mutual correlation;
        - splitting dataset into training, testing and validation.
        """
        self.dataset = pd.read_csv(path)      
        self.target_name = target_name
        self.columns_to_droop = columns_to_droop
        self.numeric_features = []
        self.object_features = []
        self.bool_features = []
        self.X_train = None
        self.y_train = None       
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.y_val_test = None
        self.X_val_test = None
        self.log_file = log_file


    def log(self, message):
        with open(self.log_file, 'a') as file:
            file.write(message + '\n')


    def clean_dataset(self):
        """
        Method for preprocessing a dataset, including:
        - removing unnecessary features for modeling;
        - removing duplicates;
        - processing outliers of data;
        - separating features by type;
        - encoding of object features;
        - processing nan values;             
        - removing features with high mutual correlation;
        - splitting dataset into training, testing and validation.
        """
        self.log('================================================================================') 
        self.log(f'Shape of dataset: {self.dataset.shape}\n')
        self.log(f'Removing columns that are not needed for the model...')
        self.dataset.drop(columns=self.columns_to_droop, errors='ignore', inplace=True)
        self.log(f'Shape of dataset: {self.dataset.shape}\n')

        self.log("Processing removing nan values...") 
        self.dataset = self.dataset.dropna()
        self.log(f'Shape of dataset: {self.dataset.shape}\n')

        self.log("Removing duplicates...")
        self.dataset.drop_duplicates(inplace=True)
        self.log(f'Shape of dataset: {self.dataset.shape}\n')

        self.log("Processing outliers of data...")
        Q1, Q3 = self.dataset['price'].quantile(0.25), self.dataset['price'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 9 * IQR
        self.dataset = self.dataset[self.dataset['price'] <= upper_bound]
        self.log(f'Shape of dataset: {self.dataset.shape}\n')

        self.log("Separating features by type...")
        self.numeric_features = self.dataset.select_dtypes(include=['number']).columns.tolist()
        self.object_features = self.dataset.select_dtypes(include=['object']).columns.tolist()
        self.bool_features = self.dataset.select_dtypes(include=['bool']).columns.tolist()
        self.log(f'numeric_features: {len(self.numeric_features)}')
        self.log(f'object_features: {len(self.object_features)}')
        self.log(f'bool_features: {len(self.bool_features)}\n')  

        self.log("Removing features with high mutual correlation...")
        self.remove_high_correlation_features() 

        self.log('')                                                 
        self.log('Splitting dataset into training, testing and validation...')
        self.splitt_dataset_train_test_val()                 
                                      
            
    def remove_high_correlation_features(self, threshold=0.9):
        """
        Removes features with a high correlation above the specified threshold.
        Keeps the features with higher correlation with the target.
        """
        corr_matrix = self.dataset[self.numeric_features + self.bool_features].\
        drop(columns = self.target_name).corr(method='spearman').abs()
        
        target_corr = self.dataset[self.numeric_features + self.bool_features].\
        corrwith(self.dataset[self.target_name], method='spearman').abs()
        
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = set()
        
        for column in upper_triangle.columns:
            for row in upper_triangle.index:
                if upper_triangle.loc[row, column] > threshold:
                    print(row, column)
                    if target_corr[row] > target_corr[column]:
                        to_drop.add(column)
                    else:
                        to_drop.add(row)

        self.dataset.drop(columns=list(to_drop), inplace=True)
        self.log(f'Shape of dataset: {self.dataset.shape}')   
               
        
    def splitt_dataset_train_test_val(self):
        """
        Splitting dataset into training, testing, validation and standardization 
        """       
        X = self.dataset.drop(columns = self.target_name, axis = 1)    
        y = self.dataset[self.target_name]   
        
        self.X_train, self.X_val_test, self.y_train, self.y_val_test = \
        train_test_split(X, y, test_size=0.3, random_state=42)   
        
        self.X_val, self.X_test, self.y_val, self.y_test = \
        train_test_split(self.X_val_test, self.y_val_test, test_size=0.3, random_state=42)
        
        self.log("Shape of:") 
        self.log(f'X_train: {self.X_train.shape}, X_val_test: {self.X_val_test.shape}') 
        self.log(f'X_val: {self.X_val.shape}, X_test: {self.X_test.shape}\n')        
                             