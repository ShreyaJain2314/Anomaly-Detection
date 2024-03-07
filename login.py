import pandas as pd
import datetime
import pickle

from sklearn.ensemble import IsolationForest
# Import libraries
#from sklearn.preprocessing import OneHotEncoder
import numpy as np
import shap
from sklearn.preprocessing import StandardScaler
import argparse, os
import boto3
import json
import scipy
from sklearn import metrics
import joblib
import pickle
from io import StringIO
import ipaddress
import numpy as np
import pandas as pd
import ast
from class_file import MyClass
import hashlib
from sklearn.preprocessing import MinMaxScaler
import hashlib

    

my_object=MyClass()

if __name__ == '__main__':
    
    #Passing in environment variables and hyperparameters for our training script
    parser = argparse.ArgumentParser()
    
    #Hyperparamaters
    parser.add_argument('--estimators', type=int, default=15)
    
    #sm_model_dir: model artifacts stored here after training
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    
    args, _ = parser.parse_known_args()
    estimators   = args.estimators
    model_dir  = args.model_dir
    sm_model_dir = args.sm_model_dir
    training_dir   = args.train
    
    ############
    #Reading in data
    ############
        
    df = pd.read_csv(training_dir+'/login_data.csv',sep=',')
    df=my_object.clean_data(df)
    df=my_object.convert_username_column_to_hash(df,'username')
    df=my_object.clean_timestamp(df)
    df['ip']=df['ip'].apply(lambda x: my_object.ip_to_int(x))
    df=my_object.float_conversion(df)
    my_object.fit_model(df)
    # Save the model as a pickle file
    my_object.save_model('isolation_forest_model.pkl')
    loaded_model = MyClass.load_model('isolation_forest_model.pkl')
    anomalies=loaded_model.model.predict(df)
    df['anomaly']=anomalies
    explainer = shap.Explainer(loaded_model.model)
   
    

    model_dict = {'login_model': loaded_model, 'tree_explainer': explainer}
    joblib.dump(model_dict, os.path.join(args.sm_model_dir, "model.joblib"))
    
###########
#Model Serving
###########
    
"""
Deserialize fitted model
"""
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    
    return model 

"""
input_fn
    request_body: The body of the request sent to the model.
    request_content_type: (string) specifies the format/variable type of the request
"""
def input_fn(request_body, content_type):
    if content_type == "application/json":
       
        print('request body', request_body)
        if isinstance(request_body, dict):
            input_data = request_body["input"]
        else:
            request_body = json.loads(request_body)
            input_data = request_body["input"]
        print('request body after changes', request_body)
        print('input data', type(input_data), input_data)
        features = pd.DataFrame.from_dict(input_data)
        return features
    else:
        raise ValueError("{} not supported by script!".format(content_type))




"""
predict_fn
    input_data: returned array from input_fn above
    model (sklearn model) returned model loaded from model_fn above
"""
def predict_fn(input_data, model_dict):
    print("Inside the predict function") 
    print("Input Data: ",input_data)
    model=model_dict['login_model']
    exp=model_dict['tree_explainer']
    #model=loaded_model.model   
    input_data=model.convert_username_column_to_hash(input_data,'username')
    input_data=model.clean_timestamp(input_data)
    input_data['ip']=input_data['ip'].apply(lambda x: model.ip_to_int(x))
    input_data=model.float_conversion(input_data)
    print(input_data)
    predictions=model.model.predict(input_data)
    #predictions_df=pd.DataFrame(predictions,columns=['prediction'])
    
        # select columns that start with 'payment_type_' and 'status_'
    shap_values = model_dict['tree_explainer'].shap_values(input_data)
    print(shap_values)
    shap_scores=pd.DataFrame(shap_values,columns=['object_pk_score','username_score','timestamp_score','ip_score'])
    print(shap_scores)
    baseline_score = shap_scores.mean(axis=1)
    
    category_labels = ['Low','High']
    categories_df = shap_scores.apply(lambda row: pd.cut(row, bins=[-np.inf, baseline_score[row.name], np.inf], labels=category_labels), axis=1)
    categories_df['predictions']=predictions
    print(categories_df)
    
    
#     abs_scores=np.abs(shap_df)
#     abs_baseline=np.abs(baseline_score)
#     scaled_scores = abs_scores / 10
#     scaled_baseline=abs_baseline/10
#     category_labels = ['Low', 'High']
#     baseline = scaled_baseline.iloc[0]  
#     print(baseline)
#     thresholds = [baseline]  
#     numeric_scores = scaled_scores.iloc[0].apply(pd.to_numeric, errors='coerce')
#     categories = pd.cut(numeric_scores, bins=[-np.inf] + thresholds + [np.inf], labels=category_labels)
#     for feature, category in zip(scaled_scores.columns, categories):
#         print(f"{feature}: {category}")

#     category_labels = ['Low', 'High']
#     thresholds = (np.abs(shap_df)/10).mean(axis=1)
#     categories_df = shap_df.apply(lambda row: pd.cut(row, bins=[-np.inf, row.mean(), np.inf], labels=category_labels), axis=1)
#     #print(categories_df)
#     categories_df['predictions']=predictions
#     print(categories_df)
    #df=pd.concat([predictions_df,shap_df],axis=1)
    #print(cat)
    #df=pd.concat([predictions_df,shap_df],axis=1)
    
    output=categories_df.to_dict(orient='records')
    return output
"""
output_fn
    prediction: the returned value from predict_fn above
    content_type: the content type the endpoint expects to be returned. Ex: JSON, string

"""

def output_fn(prediction, content_type):
    # csv_string = prediction.to_csv(index=False)
    respJSON = {'output': prediction}   
    return respJSON