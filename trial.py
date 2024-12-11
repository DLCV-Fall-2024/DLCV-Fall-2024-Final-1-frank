import json
import zipfile
import torch
from transformers import pipeline
from datasets import load_dataset

MODEL_NAME = "LLaVA-1.5-7b"
DATASET_NAME = "ntudlcv/dlcv_2024_final1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qa_pipeline = pipeline("text2text-generation", model=MODEL_NAME, device=0 if device.type == "cuda" else -1)

dataset = load_dataset(DATASET_NAME, split="test")

predictions = {}
for data in dataset:
    image = data["image"]  
    conversations = data["conversations"]  
    question = conversations["input"] 

    answer = qa_pipeline(question)[0]["generated_text"]
    predictions[data["id"]] = answer

with open("submission.json", "w") as f:
    json.dump(predictions, f, indent=4)

API_KEY = "your_api_key_here"
with open("api_key.txt", "w") as f:
    f.write(API_KEY)

with zipfile.ZipFile("pred.zip", "w") as zipf:
    zipf.write("submission.json")
    zipf.write("api_key.txt")

print("Genration finished!")