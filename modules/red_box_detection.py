import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
from ultralytics import YOLO
from torchvision.transforms import functional as F
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from depth_anything_v2.dpt import DepthAnythingV2 

class CropRedBoxModel():
    def __init__(self, 
                vit_finetune_path="checkpoints/vit_detection/vit_finetuned.pth",
                depth_model_path="checkpoints/depth_anything/depth_anything_v2_vitl.pth", 
                categories_path="checkpoints/vit_detection/categories.json", 
                yolo_path="checkpoints/vit_detection/best.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_model = self._load_depth_anything_model(model_path=depth_model_path)
        self.vit_model = self._load_vit_model(model_path=vit_finetune_path, num_classes=43)
        self.id_to_name_mapping = self._load_categories_mapping(categories_path)
        self.yolo_model = YOLO(yolo_path).to(self.device)
        self.vit_transforms = self._get_vit_transforms()
    
    def _load_vit_model(self, model_path, num_classes):
        from transformers import ViTForImageClassification
        model = ViTForImageClassification.from_pretrained("google/vit-large-patch16-224-in21k", num_labels=num_classes)
        model.classifier = torch.nn.Linear(model.config.hidden_size, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    # -----------------------
    # Load DepthAnything Model
    # -----------------------
    def _load_depth_anything_model(self, model_path):
        model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    # -----------------------
    # Create Category Mapping
    # -----------------------
    def _load_categories_mapping(self, categories_json_path):
        with open(categories_json_path, "r") as f:
            categories_data = json.load(f)
        
        id_to_name = {i: cat["name"] for i, cat in enumerate(categories_data["categories"])}
        return id_to_name

    # -----------------------
    # Transform for ViT
    # -----------------------
    def _get_vit_transforms(self):
        return Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    # -----------------------
    # Compute Depth Information
    # -----------------------
    def calculate_depth_for_boxes(self, depth_map, boxes):
        depth_results = []
        depth_map_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255
        depth_map_normalized = depth_map_normalized.astype(np.uint8)

        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            box_depth = depth_map_normalized[y_min:y_max, x_min:x_max]

            if box_depth.size == 0:
                depth_results.append({"depth_category": "unknown", "avg_intensity": None})
                continue

            avg_intensity = np.mean(box_depth)

            if avg_intensity < 51:
                depth_category = "faraway"
            elif avg_intensity < 102:
                depth_category = "longer distance"
            elif avg_intensity < 153:
                depth_category = "mid length"
            elif avg_intensity < 204:
                depth_category = "short distance"
            else:
                depth_category = "immediate"

            depth_results.append({
                "depth_category": depth_category,
                "avg_intensity": float(avg_intensity)
            })
        return depth_results
    
    def get_red_box(self, image:Image, image_id:str) -> dict:
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        depth_map = self.depth_model.infer_image(img_bgr)

        results = self.yolo_model.predict(source=img_bgr, conf=0.3)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        if len(boxes) == 0:
            print(f"No box detected for {image_id}.")
            return None

        box = boxes[0]
        x_min, y_min, x_max, y_max = box[:4].astype(int)

        black_img = np.zeros_like(img_bgr)
        black_img[y_min:y_max, x_min:x_max] = img_bgr[y_min:y_max, x_min:x_max]
        cropped_img = black_img[y_min:y_max, x_min:x_max]
        cropped_img_for_vit = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)  # Convert back to RGB
        cropped_pil = Image.fromarray(cropped_img_for_vit)
        cropped_tensor = self.vit_transforms(cropped_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.vit_model(cropped_tensor)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            predicted_class_idx = torch.argmax(probs, dim=1).item()
            predicted_label = self.id_to_name_mapping.get(predicted_class_idx, "unknown")
            confidence = probs[0, predicted_class_idx].item()

        depth_info_list = self.calculate_depth_for_boxes(depth_map, [box])
        depth_info = depth_info_list[0]

        result_json = {
            "image_id": image_id,
            "box": [float(x_min), float(y_min), float(x_max), float(y_max)],
            "depth_category": depth_info["depth_category"],
            "avg_intensity": depth_info["avg_intensity"],
            "predicted_label": predicted_label,
            "confidence": confidence
        }
        return {"segmented_image":Image.fromarray(black_img), "detection_information":result_json}

if __name__ == "__main__":

    OUTPUT_DIR = "data/preprocess/regional_cropped_data"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model = CropRedBoxModel()
    
    dataset = load_dataset("ntudlcv/dlcv_2024_final1", split="train", streaming=True)
    
    for data in tqdm(dataset, desc="Processing images", unit="img"):
        image_id = data["id"]
        if "regional" not in image_id.lower():
            continue

        pil_img = data["image"]
        red_box_info = model.get_red_box(image=pil_img, image_id=image_id)
        if red_box_info:
            output_image_path = os.path.join(OUTPUT_DIR, f"{image_id}_output.jpg")
            cv2.imwrite(output_image_path, red_box_info["segmented_image"])
            print(f"Saved cropped image to {output_image_path}")

            json_path = os.path.join(OUTPUT_DIR, f"{image_id}_info.json")
            with open(json_path, "w") as f:
                json.dump(red_box_info["detection_information"], f, indent=4)

            print(f"Saved JSON info to {json_path}")