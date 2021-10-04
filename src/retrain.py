from preprocessing import PreProcessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_squared_error,
    precision_score,
    recall_score,
)

import warnings
warnings.filterwarnings("ignore") 
import pickle
from joblib import dump, load

class Train:
    def __init__(self):
        self.preprocesing = PreProcessing()
        self.X_train, self.y_train, self.X_test, self.y_test = self.preprocesing.run_preprocessing()
        self.model = RandomForestClassifier()
        
    def run(self):
        self.model.fit(self.X_train, self.y_train)
        dump(self.model, 'models/finalized_model.joblib') 
        
        metrics = self.show_results()
        
        return metrics
    
    def show_results(self):
        """
        Evaluete the model returning a dict of metrics.

        :parameter model: model that will be evalueted
        :parameter x_test: the test dataset
        :parameter y_test: the test target
        """
        results = dict()
        y_pred = self.model.predict(self.X_test)

        results["Model_Name"] = self.model.__class__.__name__

        acc = accuracy_score(self.y_test, y_pred)
        results["Accuracy"] = round(acc, 4)

        recall = recall_score(self.y_test, y_pred)
        results["Recall"] = round(recall, 4)

        precision = precision_score(self.y_test, y_pred)
        results["Precision"] = round(precision, 4)

        f1 = f1_score(self.y_test, y_pred)
        results["F1_score"] = round(f1, 4)

        mse = mean_squared_error(self.y_test, y_pred)
        results["MSE"] = round(mse, 4)

        # results["Class_report"] = classification_report(y_test, y_pred, labels=None)

        y_pred_proba = self.model.predict_proba(self.X_test)
        ll = log_loss(self.y_test, y_pred_proba)
        results["Log_Loss"] = format(ll)

        return results