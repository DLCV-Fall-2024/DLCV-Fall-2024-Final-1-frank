import os, json
from PIL import Image
from DINO_with_labels import process_and_save_results, load_groundingdino_model, load_depth_anything_model

PATH = "DINO_with_information_results"
ANNOT_PATH = "data/train/annotation.json"
IMAGE_PATH = "data/train/images"
OUTPUT_DIR = "data/DINO_with_depth_map"

with open(ANNOT_PATH, 'r') as f:
    data =json.load(f)

json_files = [ele for ele in os.listdir(PATH) if ele.endswith(".json")]
results = {}


processor, groundingdino_model, device = load_groundingdino_model()
depth_model = load_depth_anything_model(device=device)
for ele in data:
    if f'{ele["id"]}_detections.json' in json_files:
        with open(os.path.join(PATH, f'{ele["id"]}_detections.json'), 'r') as f:
            detections = json.load(f)
        results[ele["id"]] = detections
        continue
    else:
        image = Image.open(os.path.join(IMAGE_PATH, f'{ele["id"]}.png'))
        info = process_and_save_results(image, ele["id"], processor, groundingdino_model, depth_model, device, OUTPUT_DIR)
        results[ele["id"]] = info
with open(os.path.join(OUTPUT_DIR, "regional_coord.json"), 'w') as f:
    json.dump(results, f, indent=4)