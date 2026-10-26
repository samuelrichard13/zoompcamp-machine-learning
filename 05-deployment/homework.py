import pickle
from pathlib import Path
<<<<<<< HEAD
from model import normalize_record

MODEL_PATH = Path(__file__).with_name("pipeline.bin")
CLIENT = {
  "lead_source": "paid_ads",
  "industry": "technology",
  "employment_status": "employed",
  "location": "north_america",
  "number_of_courses_viewed": 2,
  "annual_income": 79276.0,
  "interaction_count": 4,
  "lead_score": 0.41
}


with MODEL_PATH.open("rb") as model_file:
    pipeline = pickle.load(model_file)

record = normalize_record(CLIENT)
first = float(pipeline.predict_proba([record])[0, 1])
second = float(pipeline.predict_proba([record])[0, 1])

assert 0 < first < 1
assert first == second
print({"conversion_probability": round(first, 3), "conversion": first >= 0.5})

=======

model_file = "https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/33ad5e9ad4d56fcfdd8ff162eca75b7628c8d1ed/cohorts/2026/homework/05-deployment/pipeline.bin"


MODEL_PATH = Path("https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/05-deployment/pipeline.bin")
with MODEL_PATH.open("rb") as model_file:
    pipeline = pickle.load(model_file)
>>>>>>> 2863316 (test)
