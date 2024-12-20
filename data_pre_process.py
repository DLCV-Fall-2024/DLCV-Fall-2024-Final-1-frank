import os
import json
import cv2
import glob

image_dir = "./data_preparation/"
output_image_dir = "./dataset/images/train/"  
output_label_dir = "./dataset/labels/train/"
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

image_files = glob.glob(os.path.join(image_dir, "*.jpg"))

for img_path in image_files:
    base_name = os.path.splitext(os.path.basename(img_path))[0]  # e.g. "Train_general_0_bbox_0"

    json_candidates = glob.glob(os.path.join(image_dir, f"{base_name}*.json"))
    if len(json_candidates) == 0:
        continue
    json_path = json_candidates[0]

    img = cv2.imread(img_path)
    if img is None:
        continue
    H, W, C = img.shape

    with open(json_path, "r") as f:
        data = json.load(f)
    box = data["box"]
    x_min, y_min, x_max, y_max = box
    
    x_center = ((x_min + x_max) / 2) / W
    y_center = ((y_min + y_max) / 2) / H
    width = (x_max - x_min) / W
    height = (y_max - y_min) / H
    class_id = 0  

    cv2.imwrite(os.path.join(output_image_dir, os.path.basename(img_path)), img)

    with open(os.path.join(output_label_dir, base_name + ".txt"), "w") as f:
        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
