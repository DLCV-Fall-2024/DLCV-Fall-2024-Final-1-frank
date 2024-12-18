import cv2
import numpy as np
import os
import json

# Function to process images using bounding box information from JSON
def process_images_with_bounding_boxes(image_dir, json_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all images in the image directory
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)
            json_path = os.path.join(json_dir, filename.replace('_annotated.jpg', '_detections.json').replace('_annotated.png', '_detections.json'))
            output_path = os.path.join(output_dir, filename)

            # Check if corresponding JSON file exists
            if not os.path.exists(json_path):
                print(f"JSON file not found for {filename}. Skipping...")
                continue

            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to load image {filename}. Skipping...")
                    continue

                # Load bounding box information from JSON
                with open(json_path, 'r') as f:
                    detections = json.load(f)

                # Create a black mask
                mask = np.zeros(image.shape[:2], dtype=np.uint8)

                # Draw bounding boxes on the mask
                for detection in detections:
                    box = detection['box']
                    x_min, y_min, x_max, y_max = map(int, box)
                    cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)

                # Apply the mask to the image
                result = image.copy()
                result[mask == 0] = [0, 0, 0]

                # Save the processed image
                cv2.imwrite(output_path, result)
                print(f"Processed and saved: {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

# Directories
image_directory = "./DINO_with_information_results/"
json_directory = "./DINO_with_information_results/"
output_directory = "./processed_DINO_results/"

# Process images
process_images_with_bounding_boxes(image_directory, json_directory, output_directory)
