import os
import cv2
import json
import torch
import numpy as np
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torchvision
from torchvision.ops import nms

# Define directories
OUTPUT_DIR = "data_preparation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_coda_dataset(split="train"):
    dataset = load_dataset("ntudlcv/dlcv_2024_final1", split=split, streaming=True)
    return dataset

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

def draw_single_box_on_image(image, box, color=(255, 0, 0), thickness=2):
    x_min, y_min, x_max, y_max = map(int, box)
    image_copy = image.copy()
    cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), color, thickness)
    return image_copy

def save_single_detection_result(image_id, box, label, output_dir):
    json_result = {
        "class": str(label),
        "box": [float(coord) for coord in box]
    }
    json_path = os.path.join(output_dir, f"{image_id}_single_{int(box[0])}_{int(box[1])}.json")
    with open(json_path, "w") as f:
        json.dump(json_result, f, indent=4)
    print(f"Saved single detection result to {json_path}")

def process_and_save_results(image, image_id, processor, groundingdino_model, device, output_dir):
    image_np = np.array(image)
    if image_np.ndim == 3 and image_np.shape[-1] == 3:
        pass
    else:
        raise ValueError("Input image is not in the expected format (H x W x 3).")

    boxes, _, labels = get_bounding_boxes(image_np, processor, groundingdino_model, device)
    
    if len(boxes) == 0:
        print(f"No objects detected in image ID: {image_id}")
        return

    for i, (box, label) in enumerate(zip(boxes, labels)):
        single_image = draw_single_box_on_image(image_np, box)
        annotated_image_path = os.path.join(output_dir, f"{image_id}_bbox_{i}.jpg")
        cv2.imwrite(annotated_image_path, cv2.cvtColor(single_image, cv2.COLOR_RGB2BGR))
        print(f"Saved single bbox image to {annotated_image_path}")

        save_single_detection_result(image_id + f"_bbox_{i}", box, label, output_dir)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_coda_dataset(split="train")
    processor, groundingdino_model, device = load_groundingdino_model()

    for data in dataset:
        image = data["image"]  # PIL Image
        image_id = data["id"]  # Unique ID for the image

        if "general" in image_id.lower():
            print(f"Processing image ID: {image_id}")
            process_and_save_results(image, image_id, processor, groundingdino_model, device, OUTPUT_DIR)
        else:
            print(f"Skipping image ID: {image_id}")
