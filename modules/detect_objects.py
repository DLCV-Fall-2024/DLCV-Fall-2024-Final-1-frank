import os
import cv2
import json
import torch
import numpy as np
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from depth_anything_v2.dpt import DepthAnythingV2
from torchvision.ops import nms
from tqdm import tqdm

from utils import load_groundingdino_model

class DetectObjectModel():
    def __init__(self, 
                depth_model_path="checkpoints/depth_anything/depth_anything_v2_vitl.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_model = self._load_depth_anything_model(depth_model_path=depth_model_path)
        self.grounding_dino_processor, self.grounding_dino_model, self.grounding_dino_device = load_groundingdino_model(device=self.device)
        self.text_prompt = "pedestrian. car. truck. motorcycle. bicycle. umbrella. fire hydrant. \
            train. stop sign. bench. suitcase. traffic light. stop-sign. bus. dog. chair. cone. person. \
                backpack. sign. bottle. clock. parking. potted plant. airplane. cow. horse. skateboard. \
                    handbag. parking meter. boat. bird. horse. tv. dining table. toilet. sports ball. kite. \
                        sheep. cat. refrigerator. elephant. frisbee. bed. oven. bear. teddy bear. mouse. book. \
                            laptop. couch. snowboard. cup. bowl. spoon. vase. cell phone. banana. tennis racket. \
                                keyboard. surfboard. toothbrush. microwave. fence. bridge. tunnel. protecting facilities. \
                                    rail. guard rail. ditch. slump. steep. turn. signal. traffic signal. ramp. trailer. tractor. \
                                        van. vehicle. taxi. bus station. animal. stroller. walkers. segways. roadblocks. barricades. \
                                            construction machinery. fallen tree. potholes. rocks. flood. mud. ambulance. \
                                                traffic island. roundabout. speed bump. tool booth. ladder."
    
    def _load_depth_anything_model(self, depth_model_path):
        model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
        model.load_state_dict(torch.load(depth_model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def get_bounding_boxes(self, image):

        inputs = self.grounding_dino_processor(images=image, text=self.text_prompt, return_tensors="pt").to(self.grounding_dino_device)
        with torch.no_grad():
            outputs = self.grounding_dino_model(**inputs)
        results =  self.grounding_dino_processor.post_process_grounded_object_detection(
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

    def calculate_depth_for_boxes(self, depth_map, boxes):
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

    def draw_boxes_on_image(self, image, boxes, color=(255, 0, 0), thickness=2):
        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        return image

    def get_objs_info(self, image, image_id, output_dir:str=None):
        # Ensure the image is in NumPy format and valid for DepthAnything
        image_np = np.array(image)
        if image_np.ndim == 3 and image_np.shape[-1] == 3:
            raw_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("Input image is not in the expected format (H x W x 3).")

        # Step 1: Generate depth map
        depth_map = self.depth_model.infer_image(raw_image)

        # Step 2: Use Grounding DINO to detect objects
        boxes, _, labels = self.get_bounding_boxes(image_np)
        
        if len(boxes) == 0:
            print(f"No objects detected in image ID: {image_id}")
            return None 
        
        # Step 3: Calculate depth categories for each bounding box
        depths = self.calculate_depth_for_boxes(depth_map, boxes)
        
        json_results = []
        for box, label, depth in zip(boxes, labels, depths):
            json_results.append({
                "class": str(label),
                "box": [float(coord) for coord in box],
                "depth_category": depth
            })
        
        if output_dir:
            # Step 4: Draw bounding boxes on image
            image_with_boxes = self.draw_boxes_on_image(image_np.copy(), boxes)

            # Step 5: Save annotated image
            annotated_image_path = os.path.join(output_dir, f"{image_id}.jpg")
            cv2.imwrite(annotated_image_path, cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
            print(f"Saved annotated image to {annotated_image_path}")
            #
            # Step6: Save detection results to JSON
            json_path = os.path.join(output_dir, f"{image_id}_detections.json")
            with open(json_path, "w") as f:
                json.dump(json_results, f, indent=4)
            print(f"Saved detection results to {json_path}")
            
        return json_results

if __name__ == "__main__":
    
    # Define directories
    OUTPUT_DIR = "data/preprocess/obj_detect_data"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model = DetectObjectModel()

    dataset = load_dataset("ntudlcv/dlcv_2024_final1", split="train", streaming=True)
    for data in tqdm(dataset, desc="Processing images", unit="img"):
        image = data["image"]  # PIL Image
        image_id = data["id"]  # Unique ID for the image
        
        info = model.get_objs_info(image, image_id, OUTPUT_DIR)