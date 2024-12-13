import os
import numpy as np
from PIL import Image
from datasets import load_dataset
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch


def load_sam_model():
    sam_checkpoint = "./models/sam_vit_h_4b8939.pth"  # Path to the SAM model checkpoint
    model_type = "vit_h"  # Model type (vit_h, vit_l, vit_b)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # Move SAM model to GPU if available
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator


def process_and_save_segmentation(dataset, mask_generator, output_dir):
    for data in dataset:
        image = data["image"]
        image_id = data["id"]
        image_np = np.array(image)

        # Generate masks using SAM Automatic Mask Generator
        masks = mask_generator.generate(image_np)

        # Combine all masks into one binary mask
        final_mask = np.zeros(image_np.shape[:2], dtype=bool)
        for mask in masks:
            final_mask |= mask["segmentation"]

        # Create a black-masked image
        overlayed_image = create_black_masked_image(image_np, final_mask)

        # Save the segmentation result
        save_path = os.path.join(output_dir, f"{image_id}_segmentation.png")
        overlayed_image.save(save_path)
        print(f"Saved segmentation for {image_id} at {save_path}")


def create_black_masked_image(original_image, mask):
    black_image = np.zeros_like(original_image)
    black_image[mask] = original_image[mask]
    return Image.fromarray(black_image)


def load_coda_dataset(split="train"):
    dataset = load_dataset("ntudlcv/dlcv_2024_final1", split=split, streaming=True)
    return dataset


if __name__ == "__main__":
    dataset = load_coda_dataset(split="train")
    mask_generator = load_sam_model()

    SEGMENTATION_OUTPUT_DIR = "SAM_all_segmentation_results"
    os.makedirs(SEGMENTATION_OUTPUT_DIR, exist_ok=True)

    process_and_save_segmentation(dataset, mask_generator, SEGMENTATION_OUTPUT_DIR)