import streamlit as st
# import pickle
import numpy as np

import joblib
df=joblib.load("df.pkl")


import pandas as pd


from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

x_train=joblib.load('x_t.joblib')
y_train=joblib.load('y_t.joblib')
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
column_trans=make_column_transformer((OneHotEncoder(),[0,1,8,11,12]),
                                    remainder='passthrough')
lin = RandomForestRegressor()
pipe = Pipeline([('a1',column_trans),('a2',lin)])
pipe.fit(x_train,y_train)
st.title('Laptop Price Predictor')

col1,col2=st.columns([1,1])

company=col1.selectbox('Brand',df['Company'].unique())

type=col2.selectbox('Type',df['TypeName'].unique())

RAM=col1.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

weight=col2.text_input('Weight of the laptop')

touchscreen=col1.selectbox('Touchscreen',['No','Yes'])

IPS=col2.selectbox('IPS',['No','Yes'])

screen_size=col1.text_input('Screen Size')

resolution=col2.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160',
'3200x1800','2880x1800','2560x1600','1560x1440','2304x1440'])

CPU=col1.selectbox('Brand',df['CPU brand'].unique())

hdd=col2.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd=col1.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

GPU=col2.selectbox('GPU',df['Gpu brand'].unique())

OS=col2.selectbox('OS',df['OS'].unique())

col1.text("press the folowing button for prediction")
if col1.button('Predict Price'):
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if IPS == 'Yes':
        IPS = 1
    else:
        IPS = 0
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/float(screen_size)
    query=[[company,type,float(screen_size),RAM,float(weight),touchscreen,IPS,ppi,CPU,hdd,ssd,GPU,OS]]
    st.title("The predicted price of this configuration is " + 
             str(int(np.exp(pipe.predict(query)))))
