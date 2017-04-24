
# coding: utf-8

# In[ ]:

from __future__ import absolute_import


# In[5]:

get_ipython().system(' sudo apt-get update')


# In[4]:

get_ipython().system(' sudo apt-get install openjdk-8-jre -y')


# In[7]:

get_ipython().system(' pip install h2o')


# In[8]:

import os
from boto.s3.connection import S3Connection
from boto.s3.key import Key
import h2o
import numpy as np
import pandas as pd
from tabulate import tabulate
# initialize the model scoring server
h2o.init()


# In[9]:

# function to get files from s3
def pull_file_from_s3(key):
    def get_bucket():            
        access= os.environ['SECRET_ENV_AWS_ACCESS_KEY_BRETT'] 
        secret= os.environ['SECRET_ENV_AWS_SECRET_KEY_BRETT']
        customer = 'demonstration'
        conn = S3Connection(access,secret)
        b = conn.get_bucket('dsclouddata',validate=False)
        return b

    s3_bucket = get_bucket()
    payload = s3_bucket.get_key(key)
    local_file = payload.get_contents_to_filename(key)
    return key

    
# download the model from s3
downloaded_model = pull_file_from_s3('/home/jupyter/Predict-Churn/GBM_RF_ensemble_')  


# In[11]:

def churn_predict(State,AccountLength,AreaCode,Phone,IntlPlan,VMailPlan,VMailMessage,DayMins,DayCalls,DayCharge,EveMins,EveCalls,EveCharge,NightMins,NightCalls,NightCharge,IntlMins,IntlCalls,IntlCharge,CustServCalls):
    # connect to the model scoring service
    h2o.init()

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


# In[12]:

State = "KS"
AccountLength = 1 
AreaCode = 213
Phone = "362-1234"
IntlPlan = "no"
VMailPlan = "no"
VMailMessage = 0
DayMins = 0
DayCalls = 2
DayCharge = 20
EveMins = 120
EveCalls = 97
EveCharge = 7
NightMins = 2
NightCalls = 7
NightCharge = 10
IntlMins = 13
IntlCalls = 0
IntlCharge = 3.67
CustServCalls = 2
    

churn_predict(State,AccountLength,AreaCode,Phone,IntlPlan,VMailPlan,VMailMessage,DayMins,DayCalls,DayCharge,EveMins,EveCalls,EveCharge,NightMins,NightCalls,NightCharge,IntlMins,IntlCalls,IntlCharge,CustServCalls)

