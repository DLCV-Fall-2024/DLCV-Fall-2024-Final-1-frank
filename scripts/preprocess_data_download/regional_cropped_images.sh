
gdown "https://drive.google.com/uc?id=1rI73YH4LtIPOL3Jz3R49JaWZfNRE2_ZU" 
unzip regional_results.zip -d data/preprocess/ -x "__MACOSX*" "*/.DS_Store"
rm regional_results.zip
mv data/preprocess/regional_results data/preprocess/regional_cropped_data