# Pillow==9.5.0
from datasets import load_dataset
import cv2
import numpy as np

dataset = load_dataset(
    "ntudlcv/dlcv_2024_final1"
    # streaming=True
)
dataset.save_to_disk(r"C:\Users\austi\NoBackup\DLCV\final\dataset")
# dataset_val = load_dataset(
#     "ntudlcv/dlcv_2024_final1", split="val"
# )
# dataset_test = load_dataset(
#     "ntudlcv/dlcv_2024_final1", split="test"
# )

# for data in dataset:
#     img = cv2.cvtColor(np.array(data["image"]), cv2.COLOR_RGB2BGR)
#     cv2.imwrite("image.png", img)
#     break
