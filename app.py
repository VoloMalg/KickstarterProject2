#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# API: PRODUCTION PART

# from model_train import FEATURE_LS

import pickle as pkl
import pandas as pd
from flask import Flask, request, jsonify, render_template
app = Flask(__name__) #Initialize the flask App


MODEL_FILE = "model.pkl"
FEATURE_LS = ['main_category_encoded',
              'category_encoded',
              'usd_goal_real',
              "duration_days",
              "launch_month",
              'launch_day',
              'US'
             ]


# import model
# main_category_encoder, category_encoder, clf = pkl.load(MODEL_FILE)

with open(MODEL_FILE,'rb') as f:
    main_category_encoder, category_encoder, clf = pkl.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST']) 
def predict():
    
    init_features = [x for x in request.form.values()]
       
    df = pd.DataFrame(init_features,  
                      ['main_category',
                               'category',
                               'usd_goal_real',
                               "duration_days",
                               "launch_month",
                               'launch_day',
                               'US'
                              ]
                     ).T

    df['usd_goal_real'] = df['usd_goal_real'].astype(float)
    df['duration_days'] = df['duration_days'].astype(int)
    df['launch_month'] = df['launch_month'].astype(int)
    df['launch_day'] = df['launch_day'].astype(int)
    df['US'] = df['US'].astype(int)

    df["main_category_encoded"] = main_category_encoder.transform(X=df["main_category"])
    df["category_encoded"] = category_encoder.transform(X=df["category"])

    X_pred = df[FEATURE_LS]

    y_pred = clf.predict(X_pred)



    return render_template('index.html', prediction_text='Predicted status of the project is {}'.format(str(y_pred)))


if __name__ == "__main__":
    app.run()

