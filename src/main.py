from typing import Optional, Any, Dict, AnyStr, List, Union

from fastapi import FastAPI, Request
from retrain import Train
from predictor import Predictor

app = FastAPI()

JSONObject = Dict[AnyStr, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]

@app.get("/retrain")
def read_root():
    return Train().run()

@app.get("/predict/{index}")
def predict_item(index: int):
    return {"predict_model": str(Predictor().predict(index)[0]) }

@app.get("/predict/")
def predict_example():  
    return {"predict_model" : str(Predictor().predict_example()[0] ) }
