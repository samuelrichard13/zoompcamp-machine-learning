import pickle
from pathlib import Path

model_file = "https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/33ad5e9ad4d56fcfdd8ff162eca75b7628c8d1ed/cohorts/2026/homework/05-deployment/pipeline.bin"


MODEL_PATH = Path("https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/05-deployment/pipeline.bin")
with MODEL_PATH.open("rb") as model_file:
    pipeline = pickle.load(model_file)
