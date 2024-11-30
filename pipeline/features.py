import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from catboost import CatBoostRegressor


def continuous2interval(df, df_target, percent_interval=0.1):
    special_target = []
    interval_target = []
    begin = False
    temp_percent = 0
    target_distribution = df[df_target].value_counts(normalize=True).reset_index()
    target_distribution.columns = [df_target, 'proportion']
    for index, row in target_distribution.sort_values(by=df_target).iterrows():
        if row['proportion'] >= percent_interval:
            special_target.append(row[df_target])
        else:
            temp_percent += row['proportion']
            if not begin:
                begin = row[df_target]
            if temp_percent >= percent_interval:
                interval_target.append([begin, row[df_target]])
                begin = False
                temp_percent = 0
    if begin:
        interval_target.append([begin, np.inf])
    return interval_target, special_target


class FeaturesSelection:
    """
    Class for:
    - features selection based on the population stability index PSI;
    - features selection by their importance using the SHAP method.
    """
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, numeric_features, object_features, bool_features, 
                 target_name, relative_importance_threshold = 0.01, log_file='output_catboost.txt'):        
        self.PSI = None
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.relative_importance_threshold = relative_importance_threshold
        self.numeric_features = numeric_features
        self.object_features = object_features 
        self.bool_features = bool_features
        self.target_name = target_name
        self.log_file = log_file
        self.model = None


    def log(self, message):
        with open(self.log_file, 'a') as file:
            file.write(message + '\n')   

 
    def features_selection(self):   
        self.log("Processing features selection based on the population stability index PSI...")
        tmp1 = set(self.X_train.columns)
        if self.target_name in self.numeric_features:
            self.numeric_features.remove(self.target_name)
        self.selection_by_PSI(self.X_train[self.numeric_features], self.X_val[self.numeric_features])
        self.selection_by_PSI(self.X_train[self.numeric_features], self.X_test[self.numeric_features])
        tmp2 = set(self.X_train.columns)
        self.log(f'Remote features: {tmp1 - tmp2}')            
        self.log(f'Shape of X: {self.X_train.shape}\n')

        self.log("Processing features selection by their importance...")
        tmp1 = set(self.X_train.columns)
        self.features_importance()
        tmp2 = set(self.X_train.columns)
        self.log(f'Remote features: {tmp1 - tmp2}') 
        self.log(f'Shape of X: {self.X_train.shape}\n')  

        
    def PSI_factor_analysis(self, dev, val, column):
        intervals = [-np.inf] + [i[0] for i in continuous2interval(dev, column)[0]] + [np.inf]
        dev_temp = pd.cut(dev[column], intervals).value_counts(sort=False, normalize=True)
        val_temp = pd.cut(val[column], intervals).value_counts(sort=False, normalize=True)
        epsilon = 1e-10
        dev_temp += epsilon
        val_temp += epsilon
        PSI = sum(((dev_temp - val_temp) * np.log(dev_temp / val_temp)).replace([np.inf, -np.inf], 0))
        self.PSI = PSI

        
    def selection_by_PSI(self, dev, val):        
        columns_PSI_normal = []
        for column in dev.columns:
            self.PSI_factor_analysis(dev, val, column)           
            if self.PSI < 0.2:                
                columns_PSI_normal.append(column)
        self.numeric_features = columns_PSI_normal
        self.X_train = self.X_train[columns_PSI_normal + self.object_features + self.bool_features]
        self.X_val = self.X_val[columns_PSI_normal + self.object_features + self.bool_features]
        self.X_test = self.X_test[columns_PSI_normal + self.object_features + self.bool_features]
        
              
    def features_importance(self):       
        self.model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, verbose=100)
        self.model.fit(self.X_train, self.y_train, cat_features=self.object_features)

        explainer = shap.TreeExplainer(self.model, feature_perturbation="tree_path_dependent")
        shap_values = explainer(self.X_train)
        mean_abs_shap_values = np.abs(shap_values.values).mean(axis=0)
        importance = pd.DataFrame({'feature': self.X_train.columns, 'importance': mean_abs_shap_values}) \
            .sort_values(by='importance', ascending=False)
       
        shap.initjs()
        plt.figure()
        shap.summary_plot(shap_values, self.X_train, show=False)
        plt.savefig('shap_summary_plot_catboost.png', bbox_inches='tight')
        plt.close()

        relative_importance_threshold = self.relative_importance_threshold  
        max_importance = importance['importance'].max()
        self.top_important = importance[importance['importance'] >= relative_importance_threshold * max_importance]['feature']
               
        self.X_train = self.X_train[self.top_important]
        self.X_val = self.X_val[self.top_important]
        self.X_test = self.X_test[self.top_important]
        
        self.object_features = self.X_train.select_dtypes(include=['object']).columns.tolist()
       