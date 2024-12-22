import os
import cv2
import json
import torch
import numpy as np
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from depth_anything_v2.dpt import DepthAnythingV2
import torchvision
from torchvision.ops import nms

# Define directories
OUTPUT_DIR = "data/DINO_with_depth_map"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Depth Anything V2 model
def load_depth_anything_model(device):
    model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
    model.load_state_dict(torch.load("checkpoints/depth_anything/depth_anything_v2_vitl.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

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

def get_bounding_boxes(image, processor, model, device):
    text_prompt = "pedestrian. car. truck. motorcycle. bicycle. umbrella. fire hydrant. train. stop sign. bench. suitcase. traffic light. stop-sign. bus. dog. chair. cone. person. backpack. sign. bottle. clock. parking. potted plant. airplane. cow. horse. skateboard. handbag. parking meter. boat. bird. horse. tv. dining table. toilet. sports ball. kite. sheep. cat. refrigerator. elephant. frisbee. bed. oven. bear. teddy bear. mouse. book. laptop. couch. snowboard. cup. bowl. spoon. vase. cell phone. banana. tennis racket. keyboard. surfboard. toothbrush. microwave. fence. bridge. tunnel. protecting facilities. rail. guard rail. ditch. slump. steep. turn. signal. traffic signal. ramp. trailer. tractor. van. vehicle. taxi. bus station. animal. stroller. walkers. segways. roadblocks. barricades. construction machinery. fallen tree. potholes. rocks. flood. mud. ambulance. traffic island. roundabout. speed bump. tool booth. ladder."
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
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
    
    if len(boxes) > 0:
        boxes_tensor = torch.tensor(boxes).to("cpu")
        scores_tensor = torch.tensor(scores).to("cpu")
        nms_indices = nms(boxes_tensor, scores_tensor, iou_threshold=0.8).numpy()
        boxes = boxes[nms_indices]
        scores = scores[nms_indices]
        labels = labels[nms_indices]
    
    return boxes, scores, labels

def calculate_depth_for_boxes(depth_map, boxes):
    depth_results = []

    # Normalize depth map to [0, 255]
    depth_map_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255
    depth_map_normalized = depth_map_normalized.astype(np.uint8)

    # print(f"Normalized Depth map range: min={np.min(depth_map_normalized)}, max={np.max(depth_map_normalized)}")

    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        box_depth = depth_map_normalized[y_min:y_max, x_min:x_max]

        if box_depth.size == 0:
            depth_results.append({"depth_category": "unknown", "avg_intensity": None})
            continue

        avg_intensity = np.mean(box_depth)
        # print(f"Box: {box}, Avg intensity: {avg_intensity:.2f}")

        # Define thresholds based on intensity
        if avg_intensity < 51:  # 0-50 intensity
            depth_category = "immediate"
        elif avg_intensity < 102:  # 51-101 intensity
            depth_category = "short distance"
        elif avg_intensity < 153:  # 102-152 intensity
            depth_category = "mid length"
        elif avg_intensity < 204:  # 153-203 intensity
            depth_category = "longer distance"
        else:  # 204-255 intensity
            depth_category = "faraway"

        # print(f"Depth category: {depth_category}")
        depth_results.append({"depth_category": depth_category, "avg_intensity": avg_intensity})
    return depth_results

def draw_boxes_on_image(image, boxes, color=(255, 0, 0), thickness=2):
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    return image

def save_detection_results(image_id, boxes, labels, depths, output_dir):
    json_results = []
    for box, label, depth in zip(boxes, labels, depths):
        json_results.append({
            "class": str(label),
            "box": [float(coord) for coord in box],
            "depth_category": depth
        })
    # json_path = os.path.join(output_dir, f"{image_id}_detections.json")
    # with open(json_path, "w") as f:
    #     json.dump(json_results, f, indent=4)
    # print(f"Saved detection results to {json_path}")
    return json_results

def process_and_save_results(image, image_id, processor, groundingdino_model, depth_model, device, output_dir):
    # Ensure the image is in NumPy format and valid for DepthAnything
    image_np = np.array(image)
    if image_np.ndim == 3 and image_np.shape[-1] == 3:
        raw_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError("Input image is not in the expected format (H x W x 3).")

    # Step 1: Generate depth map
    depth_map = depth_model.infer_image(raw_image)

    # Step 2: Use Grounding DINO to detect objects
    boxes, _, labels = get_bounding_boxes(image_np, processor, groundingdino_model, device)
    
    if len(boxes) == 0:
        print(f"No objects detected in image ID: {image_id}")
        return
    
    # Step 3: Calculate depth categories for each bounding box
    depths = calculate_depth_for_boxes(depth_map, boxes)

    # Step 4: Draw bounding boxes on image
    image_with_boxes = draw_boxes_on_image(image_np.copy(), boxes)

    # Step 5: Save annotated image
    annotated_image_path = os.path.join(output_dir, f"{image_id}.jpg")
    cv2.imwrite(annotated_image_path, cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
    print(f"Saved annotated image to {annotated_image_path}")

    # Step 6: Save detection results to JSON
    return save_detection_results(image_id, boxes, labels, depths, output_dir)
    
def process_but_no_save_results(image, image_id, processor, groundingdino_model, depth_model, device):
    # Ensure the image is in NumPy format and valid for DepthAnything
    image_np = np.array(image)
    if image_np.ndim == 3 and image_np.shape[-1] == 3:
        raw_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError("Input image is not in the expected format (H x W x 3).")

    # Step 1: Generate depth map
    depth_map = depth_model.infer_image(raw_image)

    # Step 2: Use Grounding DINO to detect objects
    boxes, _, labels = get_bounding_boxes(image_np, processor, groundingdino_model, device)
    
    if len(boxes) == 0:
        print(f"No objects detected in image ID: {image_id}")
        return
    
    # Step 3: Calculate depth categories for each bounding box
    depths = calculate_depth_for_boxes(depth_map, boxes)
    
    json_results = []
    for box, label, depth in zip(boxes, labels, depths):
        json_results.append({
            "class": str(label),
            "box": [float(coord) for coord in box],
            "depth_category": depth
        })
    
    # Step 5: Save detection results to JSON
    return json_results

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_coda_dataset(split="train")
    processor, groundingdino_model, device = load_groundingdino_model()
    depth_model = load_depth_anything_model(device)

    results = {}
    
    for data in dataset:
        image = data["image"]  # PIL Image
        image_id = data["id"]  # Unique ID for the image
        
        info = process_and_save_results(image, image_id, processor, groundingdino_model, depth_model, device, OUTPUT_DIR)
        results[image_id] = info
        # print(image_id, info)
    with open(os.path.join(OUTPUT_DIR, "regional_coord.json"), 'r') as f:
        json.dump(results, f, indent=4)