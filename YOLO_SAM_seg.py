import os
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO  

SAFETY_RISK_CLASSES = ['pedestrian', 'car', 'truck', 'motorcycle', 'bicycle', 'umbrella', 'fire hydrant', 'train', 'stop sign', 'bench', 'suitcase', 'traffic light', 'stop-sign', 'bus', 'dog', 'chair', 'cone', 'person', 'backpack', 'sign', 'bottle', 'clock', 'parking', 'potted plant', 'airplane', 'cow', 'horse', 'skateboard', 'handbag', 'parking meter', 'boat', 'bird', 'horse', 'tv', 'dining table', 'toilet', 'sports ball', 'kite', 'sheep', 'cat', 'refrigerator', 'elephant', 'frisbee', 'bed', 'oven', 'bear', 'teddy bear', 'mouse', 'book', 'laptop', 'couch', 'snowboard', 'cup', 'bowl', 'spoon', 'vase', 'cell phone', 'banana', 'tennis racket', 'keyboard', 'surfboard', 'toothbrush', 'microwave']

def load_detection_model():
    model = YOLO('yolov8x.pt').to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model

def process_and_save_segmentation(dataset, sam_predictor, detection_model, output_dir):
    for data in dataset:
        image = data["image"]
        image_id = data["id"]
        image_np = np.array(image)
        
        results = detection_model(image_np)
        
        risk_detections = [
            box for box in results[0].boxes 
            if detection_model.names[int(box.cls)] in SAFETY_RISK_CLASSES
        ]

        for box in results[0].boxes:
            if not detection_model.names[int(box.cls)] in SAFETY_RISK_CLASSES:
                SAFETY_RISK_CLASSES.append(detection_model.names[int(box.cls)])
        
        sam_predictor.set_image(image_np)
        
        final_mask = np.zeros(image_np.shape[:2], dtype=bool)
        
        for detection in risk_detections:
            x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
            width, height = x2 - x1, y2 - y1
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            input_points = np.array([
                [center_x, center_y],
                [int(x1 + width * 0.25), int(y1 + height * 0.25)],
                [int(x1 + width * 0.75), int(y1 + height * 0.75)],
            ])
            input_labels = np.ones(len(input_points), dtype=int)
            
            masks, scores, _ = sam_predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box=np.array([x1, y1, x2, y2]),
                multimask_output=False
            )
            
            if len(masks) > 0:
                best_mask = masks[0] > 0
                final_mask |= best_mask
        
        overlayed_image = create_black_masked_image(image_np, final_mask)
        
        save_path = os.path.join(output_dir, f"{image_id}_safety_segmentation.png")
        overlayed_image.save(save_path)
        # print(f"Saved safety segmentation for {image_id} at {save_path}")

def create_black_masked_image(original_image, mask):
    black_image = np.zeros_like(original_image)
    
    black_image[mask] = original_image[mask]
    
    return Image.fromarray(black_image)

def load_sam_model():
    sam_checkpoint = "./models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    predictor = SamPredictor(sam)
    return predictor

def load_coda_dataset(split="train"):
    dataset = load_dataset("ntudlcv/dlcv_2024_final1", split=split, streaming=True)
    return dataset

if __name__ == "__main__":
    dataset = load_coda_dataset(split="train")
    sam_predictor = load_sam_model()
    detection_model = load_detection_model()
    
    SEGMENTATION_OUTPUT_DIR = "YOLO_SAM_segmentation_results"
    os.makedirs(SEGMENTATION_OUTPUT_DIR, exist_ok=True)
    
    process_and_save_segmentation(
        dataset, 
        sam_predictor, 
        detection_model, 
        SEGMENTATION_OUTPUT_DIR
    )

    print(SAFETY_RISK_CLASSES)