import os
from boto.s3.connection import S3Connection
from boto.s3.key import Key
import h2o
import numpy as np
import pandas as pd
from tabulate import tabulate
# initialize the model scoring server
h2o.init(nthreads=1,max_mem_size=1, start_h2o=True, strict_version_check = False)

def churn_predict(State,AccountLength,AreaCode,Phone,IntlPlan,VMailPlan,VMailMessage,DayMins,DayCalls,DayCharge,EveMins,EveCalls,EveCharge,NightMins,NightCalls,NightCharge,IntlMins,IntlCalls,IntlCharge,CustServCalls):
    # connect to the model scoring service
    h2o.connect()

    # open the downloaded model
    ChurnPredictor = h2o.load_model(path='GBM-RF-ensemble') 

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
    prediction = predictionsOut[1][0]
    probabilityChurn = predictionsOut[1][1]
    probabilityRetain = predictionsOut[1][2]
    return "Prediction: " + str(prediction) + " |Probability to Churn: " + str(probabilityChurn) + " |Probability to Retain: " + str(probabilityRetain)