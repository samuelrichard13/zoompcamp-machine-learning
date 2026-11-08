#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Lasso
from sklearn.metrics import root_mean_squared_error
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# parameters

alpha=0.0005
output_file = f'model_sleep_score.bin'

# data preparation

df =pd.read_csv('sleep_debt_screen_time.csv')
df["blue_light_filter_on"] = df["blue_light_filter_on"].astype(int)

categorical_variable =['primary_app_category']
numerical_variable = ['total_sleep_hours',
 'sleep_efficiency_pct',
 'sleep_onset_latency_min',
 'awakenings_count',
 'rem_sleep_pct',
 'screen_time_before_bed_min',
 'age',
 'caffeine_intake_mg',
 'deep_sleep_pct',
 'screen_brightness_pct',
 'blue_light_filter_on',
 'bedtime_hour',
 'caffeine_last_hour',
 'ambient_noise_db']

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

# training 

preprocessor = make_column_transformer(
    (StandardScaler(), numerical_variable),
    (OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_variable),
)

pipeline = make_pipeline(preprocessor,Lasso(alpha=alpha))

X_train = df_train[numerical_variable + categorical_variable]
X_test = df_test[numerical_variable + categorical_variable]
y_train = df_train.next_day_fatigue_score.values
y_test = df_test.next_day_fatigue_score.values

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print (f'rmse={root_mean_squared_error(y_test, y_pred)}')

# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump(pipeline, f_out)

print(f'the model is saved to {output_file}')