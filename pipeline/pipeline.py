from pipeline.clean import Clean
from pipeline.features import FeaturesSelection
from pipeline.training import TrainingHyperparameterSelection


class Pipeline1(Clean, FeaturesSelection, TrainingHyperparameterSelection):
    def __init__(self, path, target_name, columns_to_droop, relative_importance_threshold = 0.01): 

        Clean.__init__(self,
                         path = path,
                         target_name = target_name,
                         columns_to_droop = columns_to_droop                        
                         )
        FeaturesSelection.__init__(self,                                 
                          X_train = self.X_train,
                          X_val = self.X_val,
                          X_test = self.X_test,
                          y_train = self.y_train,
                          y_val = self.y_val,
                          y_test = self.y_test,
                          numeric_features = self.numeric_features,
                          object_features = self.object_features,
                          bool_features =  self.bool_features,
                          target_name = self.target_name,
                          relative_importance_threshold = relative_importance_threshold
                          )
        TrainingHyperparameterSelection.__init__(self,
                          X_train = self.X_train,
                          X_val = self.X_val,
                          X_test = self.X_test,
                          y_train = self.y_train,
                          y_val = self.y_val,
                          y_test = self.y_test,
                          model = self.model
                          )
        
    def clean(self):
        self.clean_dataset()

    def features_sel(self):
        self.features_selection()

    def training(self):
        self.training_hyperparameter_selection()


 