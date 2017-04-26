import os
from boto.s3.connection import S3Connection
from boto.s3.key import Key
import h2o
import numpy as np
import pandas as pd
from tabulate import tabulate
# initialize the model scoring server
h2o.init(strict_version_check = FALSE)

# function to get files from s3
def pull_file_from_s3(key):
    def get_bucket():            
        access='AKIAJXTIXBAJVFOPYSLQ' 
        secret='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'
        customer = 'demonstration'
        conn = S3Connection(access,secret)
        b = conn.get_bucket('dsclouddata',validate=False)
        return b

    s3_bucket = get_bucket()
    payload = s3_bucket.get_key(key)
    local_file = payload.get_contents_to_filename(key)
    return key

    
# download the model from s3
downloaded_model = pull_file_from_s3('GBM_model_python_1490723603840_1')  

def churn_predict(State,AccountLength,AreaCode,Phone,IntlPlan,VMailPlan,VMailMessage,DayMins,DayCalls,DayCharge,EveMins,EveCalls,EveCharge,NightMins,NightCalls,NightCharge,IntlMins,IntlCalls,IntlCharge,CustServCalls):
    # connect to the model scoring service
    h2o.init(strict_version_check = FALSE)

    # open the downloaded model
    ChurnPredictor = h2o.load_model(path=downloaded_model)  

    # define a feature vector to evaluate with the model
    newData = pd.DataFrame({'State' : State,
                            'Account Length' : AccountLength,
                            'Area Code' : AreaCode,
                            'Phone' : Phone,
                            'Int\'l Plan' : IntlPlan,
                            'VMail Plan' : VMailPlan,
                            'VMail Message' : VMailMessage,
                            'Day Mins' : DayMins,
                            'Day Calls' : DayCalls,
                            'Day Charge' : DayCharge,
                            'Eve Mins' : EveMins,
                            'Eve Calls' : EveCalls,
                            'Eve Charge' : EveCharge,
                            'Night Mins' : NightMins,
                            'Night Calls' : NightCalls,
                            'Night Charge' : NightCharge,
                            'Intl Mins' :IntlMins,
                            'Intl Calls' : IntlCalls,
                            'Intl Charge' : IntlCharge,
                            'CustServ Calls' : CustServCalls}, index=[0])
    
    # evaluate the feature vector using the model
    predictions = ChurnPredictor.predict(h2o.H2OFrame(newData))
    predictionsOut = h2o.as_list(predictions, use_pandas=False)
    return predictionsOut