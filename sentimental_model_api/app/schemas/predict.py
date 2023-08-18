from typing import Any, List, Optional
import datetime
import numpy as np

from pydantic import BaseModel
#from bikeshare_model.processing.validation import DataInputSchema

class DataInputSchema(BaseModel):
    text_msg: Optional[str] 
    

class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
    
    
class PredictionResults(BaseModel):
    #errors: Optional[Any]
    #version: str
    #predictions: Optional[List[int]]
    predictions: Optional[float]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "text_msg": "This product is good"
                    }
                ]
            }
        }
