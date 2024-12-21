import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2

os.makedirs("./regional_results/", exist_ok=True)

# -----------------------
# 1) load DepthAnything model
# -----------------------
def load_depth_anything_model(device):
    model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
    model.load_state_dict(torch.load("models/depth_anything_v2_vitl.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

# -----------------------
# 2) computing Bounding Box depth information
# -----------------------
def calculate_depth_for_boxes(depth_map, boxes):
    depth_results = []
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

        if avg_intensity < 51:  # 0-50
            depth_category = "faraway"
        elif avg_intensity < 102:  # 51-101
            depth_category = "longer distance"
        elif avg_intensity < 153:  # 102-152
            depth_category = "mid length"
        elif avg_intensity < 204:  # 153-203
            depth_category = "short distance"
        else:  # 204-255
            depth_category = "immediate"

        depth_results.append({
            "depth_category": depth_category,
            "avg_intensity": float(avg_intensity)
        })
    return depth_results

# -----------------------
# 3) initialization
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "runs/detect/train/weights/best.pt"
model = YOLO(model_path).to(device)

depth_model = load_depth_anything_model(device)

# -----------------------
# 4) load datas and inference
# -----------------------
dataset = load_dataset("ntudlcv/dlcv_2024_final1", split="train", streaming=True)

for data in tqdm(dataset, desc="Processing regional images", unit="img"):
    image_id = data["id"]
    if "regional" not in image_id.lower():
        continue

    pil_img = data["image"]
    img = np.array(pil_img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    depth_map = depth_model.infer_image(img_bgr) 

    results = model.predict(source=img_bgr, conf=0.5)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) > 0:
        box = boxes[0]
        x_min, y_min, x_max, y_max = box[:4].astype(int)

        black_img = np.zeros_like(img_bgr)
        black_img[y_min:y_max, x_min:x_max] = img_bgr[y_min:y_max, x_min:x_max]

        output_name = f"./regional_results/{image_id}_output.jpg"
        cv2.imwrite(output_name, black_img)
        print(f"Saved blackened result to {output_name}")

        depths = calculate_depth_for_boxes(depth_map, [box])
        depth_info = depths[0] 

        result_json = {
            "image_id": image_id,
            "box": [float(x_min), float(y_min), float(x_max), float(y_max)],
            "depth_category": depth_info["depth_category"],
            "avg_intensity": depth_info["avg_intensity"]
        }

        json_path = f"./regional_results/{image_id}_info.json"
        with open(json_path, "w") as f:
            json.dump(result_json, f, indent=4)
        print(f"Saved JSON info to {json_path}")

    else:
        print(f"No box detected for {image_id}.")
