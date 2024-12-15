import os
import cv2
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from segment_anything import sam_model_registry, SamPredictor
import torchvision
from torchvision.ops import nms

# Define directories
SEGMENTATION_OUTPUT_DIR = "DINO_diffusion_augmentation_results"
os.makedirs(SEGMENTATION_OUTPUT_DIR, exist_ok=True)

def load_stable_diffusion():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting").to(device)
    return pipe

# Load the CODA-LM dataset
def load_coda_dataset(split="train"):
    dataset = load_dataset("ntudlcv/dlcv_2024_final1", split=split, streaming=True)
    return dataset

# Load HuggingFace GroundingDINO model
def load_groundingdino_model():
    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    return processor, model, device

# Load SAM model
def load_sam_model():
    sam_checkpoint = "models/sam_vit_h_4b8939.pth"  
    model_type = "vit_h"  # Choose model type: vit_h, vit_l, vit_b
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)
    return predictor

def get_bounding_boxes(image, processor, model, device):
    text_prompt = "car. truck. motorcycle. bicycle."
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda"):
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,
        text_threshold=0.25,
        target_sizes=[image.shape[:2]]  
    )
    
    boxes = results[0]["boxes"].cpu().numpy()
    scores = results[0]["scores"].cpu().numpy()
    labels = np.array(results[0]["labels"])
    
    print(f"Detected {len(boxes)} bounding boxes")
    
    if len(boxes) > 0:
        boxes_tensor = torch.tensor(boxes).to("cpu")
        scores_tensor = torch.tensor(scores).to("cpu")
        nms_indices = nms(boxes_tensor, scores_tensor, iou_threshold=0.8).numpy()
        boxes = boxes[nms_indices]
        scores = scores[nms_indices]
        labels = labels[nms_indices]
    
    return boxes, scores, labels

def get_segmentation_masks(image, boxes, predictor):
    predictor.set_image(image)
    masks = []
    for box in boxes:
        box = np.array(box) 
        masks_output, scores, _ = predictor.predict(box=box, multimask_output=True)
        masks.extend(masks_output)  
    return masks

def modify_with_stable_diffusion(image, boxes, masks, pipe, prompt):
    modified_image = image.copy()

    for box, mask in zip(boxes, masks):
        x_min, y_min, x_max, y_max = map(int, box)
        crop = image[y_min:y_max, x_min:x_max]
        crop_mask = mask[y_min:y_max, x_min:x_max]

        if crop.size == 0 or crop_mask.size == 0:
            continue

        crop_mask = crop_mask.astype(np.uint8) * 255

        sd_input_size = (512, 512)
        resized_crop = cv2.resize(crop, sd_input_size)
        resized_mask = cv2.resize(crop_mask, sd_input_size, interpolation=cv2.INTER_NEAREST)

        inpaint_result = pipe(
            prompt=prompt,
            image=Image.fromarray(resized_crop),
            mask_image=Image.fromarray(resized_mask)
        )
        generated_crop = np.array(inpaint_result.images[0])

        resized_generated_crop = cv2.resize(generated_crop, (x_max - x_min, y_max - y_min))

        modified_image[y_min:y_max, x_min:x_max] = resized_generated_crop

    return modified_image

def overlay_segmentation_on_image(original_image, masks):
    result_image = np.zeros_like(original_image)
    for mask in masks:
        mask = mask.astype(bool)  
        result_image[mask] = original_image[mask]  
    return result_image

def process_both_dino_and_sam_with_diffusion(image, processor, groundingdino_model, device, sam_predictor, pipe, prompt):
    boxes, _, _ = get_bounding_boxes(image, processor, groundingdino_model, device)

    if len(boxes) == 0:
        print("No objects detected.")
        return None, None

    sam_predictor.set_image(image)
    masks = []
    for box in boxes:
        mask, _, _ = sam_predictor.predict(box=np.array(box), multimask_output=False)
        masks.append(mask[0])

    modified_image = modify_with_stable_diffusion(image, boxes, masks, pipe, prompt)

    return modified_image, masks

def process_and_save_segmentation(image, image_id, processor, groundingdino_model, device, sam_predictor, pipe, prompt, output_dir):
    image_rgb = np.array(image)

    modified_image, masks = process_both_dino_and_sam_with_diffusion(
        image_rgb, processor, groundingdino_model, device, sam_predictor, pipe, prompt
    )

    if modified_image is None or masks is None:
        print(f"No objects detected in image ID: {image_id}")
        return

    overlayed_image = overlay_segmentation_on_image(modified_image, masks)

    save_path = os.path.join(output_dir, f"{image_id}_segmentation.jpg")
    Image.fromarray(overlayed_image).save(save_path)
    print(f"Saved segmentation result for image ID: {image_id} at {save_path}")

if __name__ == "__main__":
    dataset = load_coda_dataset(split="train")
    processor, groundingdino_model, device = load_groundingdino_model()
    sam_predictor = load_sam_model()
    pipe = load_stable_diffusion()

    prompt = "A car, or a truck, or a motorcycle, or a bicycle that fits the scene, high quality, detailed"

    for data in dataset:
        image = data["image"] 
        image_id = data["id"]  
        print(f"Processing image ID: {image_id}")
        process_and_save_segmentation(image, image_id, processor, groundingdino_model, device, sam_predictor, pipe, prompt, SEGMENTATION_OUTPUT_DIR)