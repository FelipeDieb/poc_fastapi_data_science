import pickle
from preprocessing import PreProcessing
from joblib import load

class Predictor:
    def __init__(self):
        #self.model = pickle.load(open('finalized_model.sav', 'rb'))
        self.model = load('models/finalized_model.joblib')
        self.preprocesing = PreProcessing()
        self.input = self.preprocesing.X_test[0:1]
        
    def predict_example(self):
        sample_tfm = self.preprocesing.format_sample(self.input)
        return self.model.predict(sample_tfm)
    
    def predict(self, index):
        if index >= self.preprocesing.X_train.shape[0]-1:
            index = 5
        sample_tfm = self.preprocesing.format_sample(self.preprocesing.X_test[index:index+1] )
        return self.model.predict(sample_tfm)
