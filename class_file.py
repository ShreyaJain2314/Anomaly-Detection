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
import hashlib
from sklearn.preprocessing import MinMaxScaler


class MyClass :
    
    def __init__(self):
        pass
    
    def clean_data(self,df):
        #df = pd.read_csv('login_data.csv')
        json_df = pd.json_normalize(df["additional_data"].apply(json.loads))
        df = df.drop(['object_id', 'action', 'changes', 'remote_addr', 'actor_id'], axis=1)
        df = df.drop(['additional_data', 'content_type_id'], axis=1)
        df = df.drop(['id'], axis=1)
        json_df = json_df.drop(['event_entity_pk', 'event_details.deviceSessionId'], axis=1)
        json_df = json_df.drop(['events'], axis=1)
        data_df = df
        data_df = data_df.join(json_df)
        data_df=data_df.rename(columns={'object_repr':'username','event_details.remote_addr':'ip'})
        data_df=data_df.drop(columns=['event_details.login'])
        data=data_df#.drop(columns=['username'])
        return data
    
    
    def convert_username_column_to_hash(self,df,column_name):
        def hash_username(username):
            hashed_username = hashlib.sha256(username.encode()).hexdigest()
            return int(hashed_username, 16)
        
        df[column_name] = df[column_name].apply(hash_username)
        scaler = MinMaxScaler()
        df[column_name] = scaler.fit_transform(df[[column_name]])
        return df
    
    
    def clean_timestamp(self,df):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        has_microseconds = any(df['timestamp'].dt.microsecond)
        # Apply transformation if microseconds are present
        if has_microseconds:
            df['timestamp'] = df['timestamp'].apply(lambda x: x.replace(microsecond=0, tzinfo=None))
        
        df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp())
        return df
        
    def ip_to_int(self,ip):
        try:
        # try to convert to IPv4 address
            return int(ipaddress.IPv4Address(ip))
        except ipaddress.AddressValueError:
        # if not IPv4, convert to IPv6 integer
            return int(ipaddress.IPv6Address(ip))
        
    def float_conversion(self,df):
        df['ip']=df['ip'].astype(float)
        df['timestamp']=df['timestamp'].astype(float)
        df['object_pk']=df['object_pk'].astype(float)
        df['username']=df['username'].astype(float)
        return df
        
    def fit_model(self, df):
        self.model = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.01).fit(df)
    
    #def predict(self, test_row):
        #return self.model.predict([test_row])
    
    def save_model(self, filepath):
        # Save the class object to a pickle file
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)

       
    @classmethod
    def load_model(cls, filepath):
        # Load the class object from a pickle file
        with open(filepath, 'rb') as file:
            model = pickle.load(file)

        return model