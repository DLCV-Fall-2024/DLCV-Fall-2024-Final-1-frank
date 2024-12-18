import os
import cv2
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from segment_anything import sam_model_registry, SamPredictor
import torchvision
from torchvision.ops import nms

# Define directories
SEGMENTATION_OUTPUT_DIR = "DINO_segmentation_results"
os.makedirs(SEGMENTATION_OUTPUT_DIR, exist_ok=True)

def create_dataloader(dataset, batch_size=16):
    return DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

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
    text_prompt = "pedestrian. car. truck. motorcycle. bicycle. umbrella. fire hydrant. train. stop sign. bench. suitcase. traffic light. stop-sign. bus. dog. chair. cone. person. backpack. sign. bottle. clock. parking. potted plant. airplane. cow. horse. skateboard. handbag. parking meter. boat. bird. horse. tv. dining table. toilet. sports ball. kite. sheep. cat. refrigerator. elephant. frisbee. bed. oven. bear. teddy bear. mouse. book. laptop. couch. snowboard. cup. bowl. spoon. vase. cell phone. banana. tennis racket. keyboard. surfboard. toothbrush. microwave. fence. bridge. tunnel. protecting facilities. rail. guard rail. ditch. slump. steep. turn. signal. traffic signal. ramp. trailer. tractor. van. vehicle. taxi. bus station. animal. stroller. walkers. segways. roadblocks. barricades. construction machinery. fallen tree. potholes. rocks. flood. mud. ambulance. traffic island. roundabout. speed bump. tool booth. ladder."
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
    
    # print(f"Detected {len(boxes)} bounding boxes")
    
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

def overlay_segmentation_on_image(original_image, masks):
    result_image = np.zeros_like(original_image)
    for mask in masks:
        mask = mask.astype(bool)  
        result_image[mask] = original_image[mask]  
    return result_image

def process_and_save_segmentation(image, image_id, processor, groundingdino_model, device, sam_predictor, output_dir):
    image_rgb = np.array(image) 

    # Get bounding boxes and labels from GroundingDINO
    boxes, scores, labels = get_bounding_boxes(image_rgb, processor, groundingdino_model, device)

    # Skip processing if no boxes are detected
    if len(boxes) == 0:
        print(f"No objects detected in image ID: {image_id}")
        return

    # Get segmentation masks from SAM
    masks = get_segmentation_masks(image_rgb, boxes, sam_predictor)

    # Overlay segmentation on the original image
    overlayed_image = overlay_segmentation_on_image(image_rgb, masks)

    # Save the overlayed segmentation result in RGB format
    save_path = os.path.join(output_dir, f"{image_id}_segmentation.jpg")
    Image.fromarray(overlayed_image).save(save_path)
    print(f"Saved segmentation result for image ID: {image_id} at {save_path}")

if __name__ == "__main__":
    # Define dataset and models
    dataset = load_coda_dataset(split="train")
    processor, groundingdino_model, device = load_groundingdino_model()
    sam_predictor = load_sam_model()

    # Process each image in the dataset
    for data in dataset:
        image = data["image"]  # PIL Image
        image_id = data["id"]  # Unique ID for the image
        print(f"Processing image ID: {image_id}")
        process_and_save_segmentation(image, image_id, processor, groundingdino_model, device, sam_predictor, SEGMENTATION_OUTPUT_DIR)