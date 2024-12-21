# DLCV Final Project

## How to run the code?
* it's feasible to utilize CUDA_VISIBLE_DEVICES to use different GPUs.

* you might need vision transformer weight for sam throughout the usage of the repository. [Here](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth&ved=2ahUKEwj9sI2KoaqKAxWhQfUHHXjJNQUQFnoECB4QAQ&usg=AOvVaw29bUYaHDECwvcL5oJ3N4Ev) is the pretrain weight of sam_vit_h_4b8939.pth. click the botton 'Here' to download for ease.

### YOLOv8n for finding bbox in regional task
First, run the following command:

    python3 data_preparation.py

* the results images and json files will be stored in ./data_preparation/ 

* each image will have just a bbox, each json file will have just a information for a bbox

Then:

    python3 data_pre_process.py

* the outputs need to be modified slightly to be as the following architecture (manually spliting the desired ratio of train/val/test):
    ```    
    dataset/
    ├─ images/
    │  ├─ train/
    │  ├─ val/
    │  └─ test/
    └─ labels/
        ├─ train/
        ├─ val/
        └─ test/
    ```
Then, run the following command for the yolo training:

    yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640

* the resulted model weight will be stored in ./runs/detect/train/weights/best.pt

Finally, to inference the model and generate desired outputs for the regional task, run the following command:

    python3 inference.py

* the results will be saved in ./regional_results/

### Grounding DINO + Depth Anything with detected labels 

    python3 DINO_with_labels.py

* make sure to first download the pretrained weights file of depth anything v2 under the path **./models/depth_anything_v2_vitl.pth**
    
    > you can download the model [here](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true). make sure to put the model in the right folder with the correct path name (depth_anything_v2_vitl.pth).

* grounding DINO pretrained model from (IDEA-Research/grounding-dino-base) will be downloaded, be aware of the memory usage.

* results will be saved in **./DINO_with_information_results/**

### Crop using Grounding DINO resulted images and json file 

    python3 crop.py

* The images and json files should be put into the same folder **./DINO_with_information_results/**, or you can modify the image_directory or json_directory in the code with your preference. 

* results will be saved in **./processed_DINO_results/**

### Segmentation using Grounding DINO + SAM

    python3 DINO_seg.py

* make sure to first download the pretrained weights file of vision transformer under the path **./models/sam_vit_h_4b8939.pth**

* grounding DINO pretrained model from (IDEA-Research/grounding-dino-base) will be downloaded, be aware of the memory usage.

* results will be saved in **./DINO_segmentation_results/**

### Segmentation using YOLOv8x + SAM

    python3 YOLO_SAM_seg.py

* make sure to first download the pretrained weights file of vision transformer under the path **./models/sam_vit_h_4b8939.pth**

* yolov8x pretrained model will be downloaded, be aware of the memory usage.

* results will be saved in **./YOLO_SAM_segmentation_results/**

### Segmentation using SAM

    python3 SAM_all.py

* make sure to first download the pretrained weights file of vision transformer under the path **./models/sam_vit_h_4b8939.pth**

* results will be saved in **./SAM_all_segmentation_results/**

### Segmentation using Deeplabv3

    python3 deeplab_seg.py

* make sure to first download the pretrained weights file of vision transformer under the path **./models/sam_vit_h_4b8939.pth**

* deeplabv3_resnet50 pretrained model will be downloaded, be aware of the memory usage.

* the code will automatically resize images to (512, 512)
 
* results will be saved in **./Deeplab_segmentation_results/**

### Using diffusion model with Grounding DINO + SAM for data-augmentation

    python3 DINO_diffusion.py

* make sure to first download the pretrained weights file of vision transformer under the path **./models/sam_vit_h_4b8939.pth**

* stable diffusion pretrained model from (runwayml/stable-diffusion-inpainting) will be downloaded, be aware of the memory usage.

* grounding DINO pretrained model from (IDEA-Research/grounding-dino-base) will be downloaded, be aware of the memory usage.
 
* results will be saved in **./DINO_diffusion_augmentation_results/**

* you can modify the prompt in main function to guide the diffusion model for further research.

## Download datasets

* using ***gdown*** might come with the problem due to the large size of the dataset despite using '&confirm=t' method in https://stackoverflow.com/questions/60739653/gdown-is-giving-permission-error-for-particular-file-although-it-is-opening-up-f

### Segmentation by YOLOv8x + SAM 
[Dataset Link (3.34 GB)](https://drive.google.com/file/d/1cZE7crqzBCXlTS4TK-MCQihlSs1e1Kja/view?usp=sharing)

    gdown "https://drive.google.com/uc?id=1cZE7crqzBCXlTS4TK-MCQihlSs1e1Kja"
    
### Segmentation by Grounding DINO + SAM
[Dataset Link (581.8 MB)](https://drive.google.com/file/d/1pzO8zgHq5im8Ae2yqu77YeYeZDFTTFFA/view?usp=sharing)

    gdown https://drive.google.com/uc?id=1pzO8zgHq5im8Ae2yqu77YeYeZDFTTFFA

### Grounding DINO + Depth anythings v2 with multiple detection boxes (general + regional) images only!
[Dataset Link (3.37 GB)](https://drive.google.com/file/d/1D6sUu0TDGfW8euHuYhm5pd-4MvJtARlD/view?usp=sharing)

    gdown "https://drive.google.com/uc?id=1D6sUu0TDGfW8euHuYhm5pd-4MvJtARlD"

### Grounding DINO + Depth anythings v2 with multiple detection boxes (general + regional) informational json files only!
[Dataset Link (24.2 MB)](https://drive.google.com/file/d/1aR2sZ5m40xXG8XCBq4TIFwEiF6cyE-4l/view?usp=sharing)

    gdown "https://drive.google.com/uc?id=1aR2sZ5m40xXG8XCBq4TIFwEiF6cyE-4l"

### Croped results using Grounding DINO's images and json files (general + regional)
[Dataset Link (1012.2 MB)](https://drive.google.com/file/d/1YCz8eTSbEHgFMKFQkx5WsAXIhFPj4CrN/view?usp=sharing)

    gdown "https://drive.google.com/uc?id=1YCz8eTSbEHgFMKFQkx5WsAXIhFPj4CrN"

### Regional cropped-results using trained-YOLOv8n (images and json files)
[Dataset Link (217.6 MB)](https://drive.google.com/file/d/1w4xNBoO1wB57qBnb4ACnE_jcbbvf9ICK/view?usp=sharing)

    gdown "https://drive.google.com/uc?id=1w4xNBoO1wB57qBnb4ACnE_jcbbvf9ICK"
