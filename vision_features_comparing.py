import torch
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, swin_b
from timm import create_model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from datasets import load_dataset
import os
from scipy.ndimage import zoom

def load_all_encoders():
    encoders = {
        "ViT": vit_b_16(pretrained=True),
        "FocalNet": create_model("focalnet_large_fl3.ms_in22k", pretrained=True),
        "PyramidViT": create_model("twins_pcpvt_large.in1k", pretrained=True),
        "SwinTransformer": swin_b(pretrained=True)
    }
    return encoders

def preprocess_image(image):
    original_size = image.size
    image = ImageOps.pad(image, (224, 224), method=Image.BICUBIC, color=(0, 0, 0))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0), original_size

def generate_heatmap(encoder, image_tensor, original_size):
    encoder.eval()
    image_tensor.requires_grad = True

    if hasattr(encoder, "forward_features"):
        features = encoder.forward_features(image_tensor)
    elif hasattr(encoder, "features"):
        features = encoder.features(image_tensor)
    elif hasattr(encoder, "forward"):
        features = encoder.forward(image_tensor)
    else:
        raise AttributeError(f"The encoder model does not have a valid method for extracting features.")

    if isinstance(features, tuple):
        features = features[-1]  

    if features.ndim == 4:  
        pooled_features = features.mean(dim=1, keepdim=True)
    else:
        pooled_features = features.mean(dim=-1, keepdim=True)
        
    pooled_features.backward(torch.ones_like(pooled_features))
        
    gradients = image_tensor.grad[0].abs().mean(dim=0).detach().cpu().numpy()
        
    heatmap = zoom(gradients, (original_size[1] / gradients.shape[0], 
            original_size[0] / gradients.shape[1]), order=3)  
        
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap

def overlay_heatmap_on_image(image, heatmap):
    plt.figure(figsize=(8, 8))
    plt.imshow(image, alpha=0.5)
    plt.imshow(heatmap, cmap="jet", alpha=0.5)  
    plt.axis("off")
    plt.tight_layout()

def visualize_results(image, heatmaps, save_path):
    plt.figure(figsize=(20, 10))
    plt.subplot(1, len(heatmaps) + 1, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    for idx, (encoder_name, heatmap) in enumerate(heatmaps.items(), start=2):
        plt.subplot(1, len(heatmaps) + 1, idx)
        plt.imshow(image, alpha=0.5)
        plt.imshow(heatmap, cmap="jet", alpha=0.5)
        plt.title(encoder_name)
        plt.axis("off")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    encoders = load_all_encoders()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name, encoder in encoders.items():
        encoder.to(device)

    dataset = load_dataset("ntudlcv/dlcv_2024_final1", split="test")
    os.makedirs("comparisons", exist_ok=True)

    for idx, data in enumerate(dataset):
        image = data["image"]
        image_tensor, original_size = preprocess_image(image)
        image_tensor = image_tensor.to(device)

        heatmaps = {}
        for encoder_name, encoder in encoders.items():
            heatmaps[encoder_name] = generate_heatmap(encoder, image_tensor, original_size)

        save_path = f"comparisons/comparison_{idx}.png"
        visualize_results(image, heatmaps, save_path)
        print(f"Saved comparison: {save_path}")

        if idx == 20:  
            break

    print("Comparisons saved in 'comparisons' folder.")