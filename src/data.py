import pandas as pd
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self):
        self.df = pd.read_csv("train_missing.csv")
        
    def infos(self):
        print("Number of rows:", self.df.shape[0], "Number of columns:", self.df.shape[1])
    
    def cols_categorical(self):
        return self.df.drop("churn", axis=1).select_dtypes(include=["object"]).columns.values.tolist()
    
    def cols_numeric(self):
        return self.df.drop("churn", axis=1).select_dtypes(include=["number"]).columns.values.tolist()
    
    def get_data_df(self):
        return self.df
    
    def update_data(self, df):
        self.df = df
    
    def get_split_data(self):
        X = self.df.drop("churn", axis=1)
        y = self.df["churn"].replace({"no": 0, "yes": 1})
        return  train_test_split(X, y, train_size=0.8, stratify=y)