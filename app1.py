import streamlit as st
import pickle
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

x_train=joblib.load('x_t.joblib')
y_train=joblib.load('y_t.joblib')
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
column_trans=make_column_transformer((OneHotEncoder(),[0,1,8,11,12]),
                                    remainder='passthrough')
lin = RandomForestRegressor()
pipe = Pipeline([('a1',column_trans),('a2',lin)])
pipe.fit(x_train,y_train)


query=[['HP','Notebook',13.3,4,1.49,0,1,165.632118,'Intel Core i5',500,0,'Intel','Windows']]
print("The predicted price of this configuration is " + 
             str(int(np.exp(pipe.predict(query)))))
