import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Define the directory to save segmentation results
SEGMENTATION_OUTPUT_DIR = "Deeplab_segmentation_results"
os.makedirs(SEGMENTATION_OUTPUT_DIR, exist_ok=True)

# ORIGINAL_IMAGES_DIR = "original_images"
# os.makedirs(ORIGINAL_IMAGES_DIR, exist_ok=True)

# Load the CODA-LM dataset
def load_coda_dataset(split="train"):
    dataset = load_dataset("ntudlcv/dlcv_2024_final1", split=split, streaming=True)
    return dataset

# Load a pre-trained DeepLabV3 model (or any other segmentation model)
from torchvision.models.segmentation import deeplabv3_resnet50

def load_segmentation_model():
    model = deeplabv3_resnet50(pretrained=True)
    model.eval()
    return model

# Preprocessing function for the images
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

# Postprocess the segmentation output
def postprocess_segmentation(output):
    # Get the most likely class for each pixel
    segmentation = torch.argmax(output["out"], dim=1).squeeze(0).cpu().numpy()
    return segmentation

# Map segmentation results to colors (for visualization)
def apply_segmentation_color_map(segmentation):
    color_map = np.array([
        [0, 0, 0],       # Background
        [128, 64, 128],  # Road
        [70, 70, 70],    # Building
        [190, 153, 153], # Pole
        [153, 153, 153], # Sidewalk
        [250, 170, 30],  # Traffic Sign
        [220, 20, 60],   # Car (additional class)
    ])
    # Ensure all values are within the valid range
    segmentation = np.clip(segmentation, 0, len(color_map) - 1)

    colored_segmentation = color_map[segmentation]
    return Image.fromarray(colored_segmentation.astype(np.uint8))

# Overlay segmentation on original image
def overlay_segmentation_on_image(original_image, segmentation):
    # Convert original image to numpy array
    original_array = np.array(original_image.resize((512, 512)))
    
    # Create a mask where segmentation is non-background
    mask = segmentation > 0

    # Apply mask to the original image
    overlayed_image = np.zeros_like(original_array)
    overlayed_image[mask] = original_array[mask]

    return Image.fromarray(overlayed_image)

# Process and save segmentation results
def process_and_save_segmentation(dataset, model, output_dir):
    for data in dataset:
        image = data["image"]
        image_id = data["id"]

        # Save original image for reference
        # original_save_path = os.path.join(ORIGINAL_IMAGES_DIR, f"{image_id}_original.png")
        # image.save(original_save_path)

        # Preprocess the image
        input_tensor = preprocess_image(image)

        # Perform segmentation
        with torch.no_grad():
            output = model(input_tensor)

        # Postprocess segmentation output
        segmentation = postprocess_segmentation(output)

        # Create overlayed segmentation image
        overlayed_image = overlay_segmentation_on_image(image, segmentation)

        # Save the overlayed segmentation result
        save_path = os.path.join(output_dir, f"{image_id}_segmentation.png")
        overlayed_image.save(save_path, format="JPEG", quality=85)

        del input_tensor, output, segmentation, overlayed_image
        torch.cuda.empty_cache()
        
        # print(f"Saved original image at {original_save_path}")
        print(f"Saved segmentation result for {image_id} at {save_path}")

if __name__ == "__main__":
    # Load dataset and model
    dataset = load_coda_dataset(split="train")
    model = load_segmentation_model()

    # Process the dataset and save segmentation results
    process_and_save_segmentation(dataset, model, SEGMENTATION_OUTPUT_DIR)