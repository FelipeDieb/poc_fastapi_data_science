import shap
import dice_ml

class ExplainableIA:
    
    def __init__(self, model):
        self.model = model