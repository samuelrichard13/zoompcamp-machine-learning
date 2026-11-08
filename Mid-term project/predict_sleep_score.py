from fastapi import FastAPI
import uvicorn
import pickle
from typing import Dict, Any
import pandas as pd

app = FastAPI(title="Sleep Score Prediction API", description="API for predicting sleep score based on user data", version="1.0.0")

with open('model_sleep_score.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

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

@app.post("/predict")
def predict(sleeper: Dict[str, Any]):

    sleeper = pd.DataFrame([sleeper])   
    prob = pipeline.predict(sleeper)
    return  {'sleep score prediction':prob[0]}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)






