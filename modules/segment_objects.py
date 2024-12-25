import os
import cv2
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from torchvision.ops import nms
from ultralytics import YOLO  
from tqdm import tqdm

from utils import load_groundingdino_model

# Load SAM model
def load_sam_model(sam_checkpoint = "checkpoints/SAM/sam_vit_h_4b8939.pth", model_type = "vit_h",):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    predictor = SamPredictor(sam)
    return predictor

class SegDino():
    def __init__(self):
        self.predictor = load_sam_model()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.grounding_dino_processor, self.grounding_dino_model, self.grounding_device = load_groundingdino_model(device=self.device)

    def get_bounding_boxes(self, image):
        text_prompt = "pedestrian. car. truck. motorcycle. bicycle. umbrella. fire hydrant. train. stop sign. bench. suitcase. traffic light. stop-sign. bus. dog. chair. cone. person. backpack. sign. bottle. clock. parking. potted plant. airplane. cow. horse. skateboard. handbag. parking meter. boat. bird. horse. tv. dining table. toilet. sports ball. kite. sheep. cat. refrigerator. elephant. frisbee. bed. oven. bear. teddy bear. mouse. book. laptop. couch. snowboard. cup. bowl. spoon. vase. cell phone. banana. tennis racket. keyboard. surfboard. toothbrush. microwave. fence. bridge. tunnel. protecting facilities. rail. guard rail. ditch. slump. steep. turn. signal. traffic signal. ramp. trailer. tractor. van. vehicle. taxi. bus station. animal. stroller. walkers. segways. roadblocks. barricades. construction machinery. fallen tree. potholes. rocks. flood. mud. ambulance. traffic island. roundabout. speed bump. tool booth. ladder."
        inputs = self.grounding_dino_processor(images=image, text=text_prompt, return_tensors="pt").to(self.grounding_device)
        with torch.no_grad(), torch.autocast(device_type="cuda"):
            outputs = self.grounding_dino_model(**inputs)
        results = self.grounding_dino_processor.post_process_grounded_object_detection(
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

    def get_segmentation_masks(self, image, boxes):
        self.predictor.set_image(image)
        masks = []
        for box in boxes:
            box = np.array(box) 
            masks_output, scores, _ = self.predictor.predict(box=box, multimask_output=True)
            masks.extend(masks_output)  
        return masks

    def overlay_segmentation_on_image(self, original_image, masks):
        result_image = np.zeros_like(original_image)
        for mask in masks:
            mask = mask.astype(bool)  
            result_image[mask] = original_image[mask]  
        return result_image

    def get_seg_image(self, image, image_id, output_dir=None):
        image_rgb = np.array(image) 

        # Get bounding boxes and labels from GroundingDINO
        boxes, scores, labels = self.get_bounding_boxes(image_rgb)

        # Skip processing if no boxes are detected
        if len(boxes) == 0:
            print(f"No objects detected in image ID: {image_id}")
            return

        # Get segmentation masks from SAM
        masks = self.get_segmentation_masks(image_rgb, boxes)

        # Overlay segmentation on the original image
        overlayed_image = self.overlay_segmentation_on_image(image_rgb, masks)

        # get the image 
        overlayed_image = Image.fromarray(overlayed_image)
        
        # Save the overlayed segmentation result in RGB format
        if output_dir:
            save_path = os.path.join(output_dir, f"{image_id}_segmentation.jpg")
            overlayed_image.save(save_path)
            print(f"Saved segmentation result for image ID: {image_id} at {save_path}")
        return overlayed_image

class SegYOLO():
    def __init__(self, yolo_path="checkpoints/YOLO/yolov8x.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(yolo_path).to(torch.device(self.device))
        self.safety_risk_classes = ['pedestrian', 'car', 'truck', 'motorcycle', 'bicycle', 'umbrella', 'fire hydrant', 'train', 'stop sign', 'bench', 'suitcase', 'traffic light', 'stop-sign', 'bus', 'dog', 'chair', 'cone', 'person', 'backpack', 'sign', 'bottle', 'clock', 'parking', 'potted plant', 'airplane', 'cow', 'horse', 'skateboard', 'handbag', 'parking meter', 'boat', 'bird', 'horse', 'tv', 'dining table', 'toilet', 'sports ball', 'kite', 'sheep', 'cat', 'refrigerator', 'elephant', 'frisbee', 'bed', 'oven', 'bear', 'teddy bear', 'mouse', 'book', 'laptop', 'couch', 'snowboard', 'cup', 'bowl', 'spoon', 'vase', 'cell phone', 'banana', 'tennis racket', 'keyboard', 'surfboard', 'toothbrush', 'microwave']
        self.predictor = load_sam_model()
    def create_black_masked_image(self, original_image, mask):
        black_image = np.zeros_like(original_image)
        black_image[mask] = original_image[mask]
        return Image.fromarray(black_image)
    
    def get_seg_image(self, image, image_id, output_dir=None):
        image_np = np.array(image)
        
        results = self.model(image_np)
        
        risk_detections = [
            box for box in results[0].boxes 
            if self.model.names[int(box.cls)] in self.safety_risk_classes
        ]

        for box in results[0].boxes:
            if not self.model.names[int(box.cls)] in self.safety_risk_classes:
                self.safety_risk_classes.append(self.model.names[int(box.cls)])
        
        self.predictor.set_image(image_np)
        
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
            
            masks, scores, _ = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box=np.array([x1, y1, x2, y2]),
                multimask_output=False
            )
            
            if len(masks) > 0:
                best_mask = masks[0] > 0
                final_mask |= best_mask
        
        overlayed_image = self.create_black_masked_image(image_np, final_mask)
        
        if output_dir:
            save_path = os.path.join(output_dir, f"{image_id}_safety_segmentation.png")
            overlayed_image.save(save_path)
            print(f"Saved safety segmentation for {image_id} at {save_path}")
        
        return overlayed_image
        
        
if __name__ == "__main__":
    # Define directories
    SEGMENTATION_OUTPUT_DIR_DINO = "data/preprocess/segmentation_DINO"
    SEGMENTATION_OUTPUT_DIR_YOLO = "data/preprocess/segmentation_YOLO"
    os.makedirs(SEGMENTATION_OUTPUT_DIR_DINO, exist_ok=True)
    os.makedirs(SEGMENTATION_OUTPUT_DIR_YOLO, exist_ok=True)

    # Define dataset and models
    dataset = load_dataset("ntudlcv/dlcv_2024_final1", split='train', streaming=True)

    # Define model
    seg_dino = SegDino()
    seg_yolo = SegYOLO()
    
    # Process each image in the dataset
    for data in tqdm(dataset, desc="Processing images", unit="img"):
        image = data["image"]  # PIL Image
        image_id = data["id"]  # Unique ID for the image
        print(f"Processing image ID: {image_id}")
        seg_dino.get_seg_image(image, image_id, SEGMENTATION_OUTPUT_DIR_DINO)
        seg_yolo.get_seg_image(image, image_id, SEGMENTATION_OUTPUT_DIR_YOLO)