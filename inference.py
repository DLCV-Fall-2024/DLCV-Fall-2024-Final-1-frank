import os
from datasets import load_dataset
import cv2
import torch
import numpy as np
from ultralytics import YOLO

os.makedirs("./regional_results/", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO('yolov8n.pt').to(device)

dataset = load_dataset("ntudlcv/dlcv_2024_final1", split="train", streaming=True)

for data in dataset:
    image_id = data["id"]
    if "regional" not in image_id.lower():
        continue

    pil_img = data["image"]
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    results = model.predict(source=img, conf=0.5) 
    boxes = results[0].boxes.xyxy.cpu().numpy() 

    if len(boxes) > 0:
        box = boxes[0]
        x_min, y_min, x_max, y_max = box[:4].astype(int)

        black_img = np.zeros_like(img)
        black_img[y_min:y_max, x_min:x_max] = img[y_min:y_max, x_min:x_max]

        output_name = f"./regional_results/{image_id}_output.jpg"
        cv2.imwrite(output_name, black_img)
        print(f"Saved result to {output_name}")
    else:
        print(f"No box detected for {image_id}.")