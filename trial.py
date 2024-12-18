import torch
import gc
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from timm import create_model
from transformers import pipeline, AutoProcessor
from datasets import load_dataset
from PIL import Image
import json
import os

class PyramidViTAdapter:
    def __init__(self, device):
        self.device = device
        self.pvt = create_model("twins_pcpvt_large.in1k", pretrained=True).to(self.device)
        self.projector = torch.nn.Linear(512, 768).to(self.device)  

    def forward(self, image_tensor):
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            features = self.pvt.forward_features(image_tensor)  
            print(f"Features shape before pooling: {features.shape}")
            global_features = features.mean(dim=1)  
            print(f"Global features shape after pooling: {global_features.shape}")
            projected_features = self.projector(global_features)
            print(f"Projected features shape: {projected_features.shape}")
            return projected_features
        return projected_features

def preprocess_image(image):
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def preprocess_question(question):
    return question.strip()

def process_coda_lm_data(pipe, processor, pvt_adapter, dataset, save_path="submission.json", batch_size=1):
    predictions = {}
    for idx, data in enumerate(dataset):
        image = data["image"]
        questions = data["conversations"]

        image_tensor = preprocess_image(image)

        with torch.no_grad():
            vision_features = pvt_adapter.forward(image_tensor)

            for q in questions:
                question = preprocess_question(q["input"])
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {"type": "image"},
                        ],
                    },
                ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

                outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 50})
                answer = outputs["generated_text"]
                predictions[data["id"]] = answer

        del image_tensor, vision_features
        torch.cuda.empty_cache()
        gc.collect()

        if idx % batch_size == 0 and idx > 0:
            with open(save_path, "w") as f:
                json.dump(predictions, f, indent=4)

    with open(save_path, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"Saved predictions to {save_path}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pvt_adapter = PyramidViTAdapter(device)
    model_id = "llava-hf/llava-1.5-7b-hf"
    pipe = pipeline("image-to-text", model=model_id, device=0 if device == "cuda" else -1, torch_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(model_id)

    dataset = load_dataset("ntudlcv/dlcv_2024_final1", split="test")

    process_coda_lm_data(pipe, processor, pvt_adapter, dataset, batch_size=1)