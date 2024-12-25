gdown "https://drive.google.com/uc?id=1cZE7crqzBCXlTS4TK-MCQihlSs1e1Kja"
unzip YOLO_SAM_segmentation_results.zip -d data/preprocess/ -x "__MACOSX*" "*/.DS_Store"
rm YOLO_SAM_segmentation_results.zip
mv data/preprocess/YOLO_SAM_segmentation_results data/preprocess/segmentation_YOLO

gdown https://drive.google.com/uc?id=1pzO8zgHq5im8Ae2yqu77YeYeZDFTTFFA
unzip DINO_segmentation_results.zip -d data/preprocess/ -x "__MACOSX*" "*/.DS_Store"
rm DINO_segmentation_results.zip
mv data/preprocess/DINO_segmentation_results data/preprocess/segmentation_DINO