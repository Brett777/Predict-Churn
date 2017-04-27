import os
from boto.s3.connection import S3Connection
from boto.s3.key import Key
import h2o
import numpy as np
import pandas as pd
from tabulate import tabulate

# initialize the model scoring server
h2o.init(nthreads=2,max_mem_size=2, start_h2o=True)

# function to upload files to s3
def upload_file_to_s3(myFile):
    def get_bucket():
        access= 'AKIAJXTIXBAJVFOPYSLQ'
        secret= '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'
        customer = 'demonstration'
        conn = S3Connection(access,secret)
        b = conn.get_bucket('dsclouddata',validate=False)
        return b
    s3_bucket = get_bucket()
    k = Key(s3_bucket)    
    k.key = myFile
    k.set_contents_from_filename(myFile)
    k.make_public()
    successMessage = "Uploaded %s to S3."%(myFile)    
    return successMessage 

# function to get files from s3
def pull_file_from_s3(key):
    def get_bucket():            
        access= 'AKIAJXTIXBAJVFOPYSLQ'
        secret= '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'
        customer = 'demonstration'
        conn = S3Connection(access,secret)
        b = conn.get_bucket('dsclouddata',validate=False)
        return b

    s3_bucket = get_bucket()
    payload = s3_bucket.get_key(key)
    local_file = payload.get_contents_to_filename(key)
    return key

    
# download the model from s3
downloaded_model = pull_file_from_s3('gbm_grid_binomial_model_1')  


def churn_predict_batch(batchFile):
    # connect to the model scoring service
    h2o.connect(verbose=False)

    # load the user-specified file
    newData = h2o.import_file(batchFile)

    # open the downloaded model
    ChurnPredictor = h2o.load_model(path=downloaded_model)  
    
    # evaluate the feature vector using the model
    predictions = ChurnPredictor.predict(newData)
    predictions = newData.cbind(predictions)
    h2o.download_csv(predictions, 'predictions.csv')
    
    upload_file_to_s3('predictions.csv')
    successMessage2 = "Predictions saved  https://s3-us-west-1.amazonaws.com/dsclouddata/home/jupyter/predictions.csv"
    return successMessage2

