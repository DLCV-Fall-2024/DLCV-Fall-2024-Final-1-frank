import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Load HuggingFace GroundingDINO model
def load_groundingdino_model(device):
    MODEL_ID = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
    return processor, model, device