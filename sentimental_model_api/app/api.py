import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from sentimental_model import __version__ as model_version
#from sentimental_model.predict import predict
#from sentimental_model_api.app import predict

from app import __version__, schemas
from app.config import settings

import numpy as np
import tensorflow as tf
import pandas as pd
#from sentimental_model import __version__ as _version
from sentimental_model.config.core import config

from sentimental_model.processing.data_manager import load_model, get_tokenizer,load_dataset,split_data,convert_text_to_int_values,cleanText,remove_stopwords

from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

from typing import Any, List

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name="Sentimental Analysis", api_version=__version__, model_version="1.0"
    )

    return health.dict()


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs) -> Any:
    """
    Bike rental count prediction with the bikeshare_model
    """
    print("input_data:",input_data)
    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    print("input_df:0:",np.array(input_df))
    data=np.array(input_df)[0]
    print ("data:",data)
    result = predict(data)
    print("result:",result, type(result))
    
    
    #results = make_prediction(input_data=input_df.replace({np.nan: None}))
    results = 0.5
    
    pred = schemas.PredictionResults(
        predictions=result
    )
    # if results["errors"] is not None:
    #     raise HTTPException(status_code=400, detail=json.loads(results["errors"]))


    return pred

def predict(test_samples):
    

    samp = pd.DataFrame(test_samples,columns = ['Text'])
    samp['Text'] = samp['Text'].apply(lambda a :  remove_stopwords(cleanText(a)))
    
    sample_arr = np.array(samp['Text'])
    
    model = load_model("sentimental_model.keras")
    
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(sample_arr)
 
    # YOUR CODE HERE
    test_samples_pro = pad_sequences(
        tokenizer.texts_to_sequences(sample_arr), 
        padding=config.model_config.padding_val, 
        maxlen=config.model_config.maxlen, 
        truncating=config.model_config.truncating_val)

    # predict
    pred = model.predict(x=test_samples_pro)
    print ("summ:")
    print(model.summary())
    print("pred::",pred[0])
    final_val=pred[0][0]
    return final_val