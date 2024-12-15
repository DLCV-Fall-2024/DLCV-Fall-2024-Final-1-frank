# DLCV Final Project

## How to run the code?
* it's feasible to utilize CUDA_VISIBLE_DEVICES to use different GPUs.

* you might need vision transformer weight for sam throughout the usage of the repository. [Here](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth&ved=2ahUKEwj9sI2KoaqKAxWhQfUHHXjJNQUQFnoECB4QAQ&usg=AOvVaw29bUYaHDECwvcL5oJ3N4Ev) is the pretrain weight of sam_vit_h_4b8939.pth. click the botton 'Here' to download for ease.

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

### Segmentation by YOLOv8x + SAM
[Dataset Link](https://drive.google.com/file/d/1cZE7crqzBCXlTS4TK-MCQihlSs1e1Kja/view?usp=sharing)

* using ***gdown*** might come with the problem due to the large size of the dataset despite using '&confirm=t' method in https://stackoverflow.com/questions/60739653/gdown-is-giving-permission-error-for-particular-file-although-it-is-opening-up-f

### Segmentation by Grounding DINO + SAM

