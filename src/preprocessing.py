from data import Data

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    MinMaxScaler,
)
import pandas as pd
import warnings
warnings.filterwarnings("ignore") 
from joblib import dump, load


class PreProcessing:
    def __init__(self):
        self.data = Data()
        self.data.infos()
        self.numeric = self.data.cols_numeric()
        self.categorical = self.data.cols_categorical()
        self.X_train, self.X_test, self.y_train, self.y_test = self.data.get_split_data()
        self.pipeline_imputer   = self.pipeline_imputation()
        self.pipeline_transform = self.pipeline_scaling()

        
    def pipeline_imputation(self): 
        
        imputer = ColumnTransformer(
        transformers=[
                ("numeric_imputer", SimpleImputer(strategy="median"), self.numeric),
                ("categorical_imputer", SimpleImputer(strategy="most_frequent"), self.categorical),
            ]
        )

        return Pipeline(steps=[("imputer", imputer)], verbose=True)
    
    def get_column_names_from_ColumnTransformer(self,column_transformer):
        col_name = []
        for (
            transformer_in_columns
        ) in (
            column_transformer.transformers_
        ):  # the last transformer is ColumnTransformer's 'remainder'
            raw_col_name = list(transformer_in_columns[2])

            if isinstance(transformer_in_columns[1], Pipeline):
                # if pipeline, get the last transformer
                transformer = transformer_in_columns[1].steps[-1][1]
            else:
                transformer = transformer_in_columns[1]

            try:
                if isinstance(transformer, MinMaxScaler):
                    names = list(transformer.get_feature_names(raw_col_name))

                elif isinstance(transformer, OrdinalEncoder) and transformer.add_indicator:
                    missing_indicator_indices = transformer.indicator_.features_
                    missing_indicators = [
                        raw_col_name[idx] + "_missing_flag"
                        for idx in missing_indicator_indices
                    ]

                    names = raw_col_name + missing_indicators

                else:
                    names = list(transformer.get_feature_names(self.categorical))

            except AttributeError as error:
                names = raw_col_name
            col_name.extend(names)

        return col_name

    def pipeline_scaling(self):
        
        # Saving the values from the columns
        col_val = []
        for col in self.categorical:
            col_val.append(list(self.data.df[col].unique()) + ["Unknown"])
            
        scaling_enconding = ColumnTransformer(
        transformers=[
            ("scaling", MinMaxScaler(), self.numeric), 
            (
                "encoding",
                OneHotEncoder(categories=col_val, handle_unknown="ignore", sparse=False),
                        self.categorical,
                    ),
                ]
            )

        pipeline_transform = Pipeline(
            steps=[
                ("scaling_enconding", scaling_enconding),
            ]
        )

        return pipeline_transform
    
    def apply_train_pipeline(self):
        #create imputation with SimpleImputer
        X_train_imp  = self.pipeline_imputer.fit_transform(self.X_train)
        X_train_imp  = pd.DataFrame(X_train_imp, columns=(self.numeric + self.categorical))
        dump(self.pipeline_imputer, 'models/pipeline_imputer.joblib') 
        #Feature Scaling and Categorical Encoding
        X_train_tfm  = self.pipeline_transform.fit_transform(X_train_imp, self.y_train)
        columns = self.get_column_names_from_ColumnTransformer(
            self.pipeline_transform.named_steps["scaling_enconding"]
        )
        self.X_train = pd.DataFrame(X_train_tfm, columns=columns)
        dump(self.pipeline_transform, 'models/pipeline_transform.joblib') 
        
    def apply_test_pipeline(self):
        #create imputation with SimpleImputer
        X_test_imp  = self.pipeline_imputer.fit_transform(self.X_test)
        X_test_imp  = pd.DataFrame(X_test_imp, columns=(self.numeric + self.categorical))
        #Feature Scaling and Categorical Encoding
        X_test_tfm  = self.pipeline_transform.fit_transform(X_test_imp, self.y_test)
        columns = self.get_column_names_from_ColumnTransformer(
            self.pipeline_transform.named_steps["scaling_enconding"]
        )
        self.X_test = pd.DataFrame(X_test_tfm, columns=columns)
        
    def run_preprocessing(self):
        print("Running pipelines on training data ... ")
        self.apply_train_pipeline()
        print("Running pipelines on test data .. ")
        self.apply_test_pipeline()
        return self.X_train, self.y_train, self.X_test, self.y_test
        
    def format_sample(self, sample_raw):
        pipeline_imputer = load('models/pipeline_imputer.joblib') 
        pipeline_transform = load('models/pipeline_transform.joblib') 
        
        sample_imp = pipeline_imputer.transform(sample_raw)
        sample_imp = pd.DataFrame(sample_imp, columns=(self.numeric + self.categorical))
        
        sample_tfm = pipeline_transform.transform(sample_imp)
        columns = self.get_column_names_from_ColumnTransformer(
            pipeline_transform.named_steps["scaling_enconding"]
        )
        
        return pd.DataFrame(sample_tfm, columns=columns)
        
