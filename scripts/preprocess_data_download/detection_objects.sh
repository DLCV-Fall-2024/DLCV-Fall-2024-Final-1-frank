
gdown "https://drive.google.com/uc?id=1i3N57RsCXXKZeHWqf8B5lzBs-pncG0oN"
unzip DINO_with_information_results.zip -d data/preprocess/ -x "__MACOSX*" "*/.DS_Store"
rm DINO_with_information_results.zip
mv data/preprocess/DINO_with_information_results data/preprocess/obj_detect_data