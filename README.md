# Notes

* All executions are performed in the **root** directory.

# Initialization

## A. Environment Setup

1. Create a new environment (python version needs to `>=3.10`), and download the required packages
    
    ```
    conda create -n <your_env_name> python=3.10 -y
    conda activate <your_env_name>
    pip install -r requirement.txt
    ```

    * If you face some error about installing ```flash-attn==2.5.8```, please comment the ```flash-attn==2.5.8``` first and conduct ```pip install -r requirements.txt``` again. Then, install ```flash-attn==2.5.8``` individually.

## B. Download pre-trained weights

* Download the weights for pre-trained model:

    ```
    bash scripts/model_weights_download.sh
    ```

## C. Data Preparation

* Conduct the following instruction

    ```
    python3 data_download.py --dataset_types test
    ```

    * The data would be downloaded in ```data``` folder

# Prediction

* There are some strategies you could choose:

    1. **(Best Choice) Strategy 1: Add Object Detection Prompts(Use only fine-tuning strategy in suggestion Task)**

    2. Strategy 2: Add Object Detection Prompts 

    3. Strategy 3: Only fine-tuning

    4. Strategy 4: Concat Object Segment Image's Tokens

* **Notes**: While conducting the prediction, you would need to be ask to **Enter your token**. Go to **HuggingFace** and apply a api token.

## (Best Choice) Strategy 1

1. If ```checkpoints/llava-v1.5-7b-lora_add_obj_info_prompt_3/``` or ```checkpoints/llava-v1.5-7b-lora_5``` not exists, download the checkpoints by conduct the following command

    ```
    bash scripts/strategy/llava-v1.5-7b-lora_split_task/download.sh
    ```

2. Conduct the prediction

    ```
    bash scripts/strategy/llava-v1.5-7b-lora_split_task/predict.sh
    ```

## Strategy 2

1. If ```checkpoints/llava-v1.5-7b-lora_add_obj_info_prompt_3/``` not exists, download the checkpoints by conduct the following command

    ```
    bash scripts/strategy/llava-v1.5-7b-lora_add_obj_info_prompt_3/download.sh
    ```

2. Conduct the prediction

    ```
    bash scripts/strategy/llava-v1.5-7b-lora_add_obj_info_prompt_3/predict.sh
    ```

## Strategy 3

1. If ```checkpoints/llava-v1.5-7b-lora_5``` not exists, download the checkpoints by conduct the following command

    ```
    bash scripts/strategy/llava-v1.5-7b-lora_5/download.sh
    ```

2. Conduct the prediction

    ```
    bash scripts/strategy/llava-v1.5-7b-lora_5/predict.sh
    ```

## Strategy 4

1. If ```checkpoints/lava-v1.5-7b-lora_add_seg_img_token_5``` not exists, download the checkpoints by conduct the following command

    ```
    bash scripts/strategy/llava-v1.5-7b-lora_add_seg_img_token_5/download.sh
    ```

2. Conduct the prediction

    ```
    bash scripts/strategy/llava-v1.5-7b-lora_add_seg_img_token_5/predict.sh
    ```

# Training

## A. Having ```Wandb``` Accounts 

* You have to create a ```wandb``` account to trace the training result [here](https://wandb.ai/)

## B. Download Pre-trained Weights

* Make sure that you have conducted ```bash model_weights_download.sh```

## C. Download Dataset 

* Conduct the following instruction

    ```
    python3 data_download.py --dataset_types train
    ```

    * (Optional)You could modify given parameters in ```--dataset_types``` to download which split of data you want to download. e.g. ```--dataset_types train``` for only download the training dataset

    * (Optional)You could also choose to add ```--max_dataset_num``` to download partial of the dataset

    * **Watch Out!!**: Download the whole ```train``` dataset might requires 4-5 hours

## D. Prepare Preprocessing Training Data

* Depend on what your strategies(If you combine **2.** **3.**, you need to download all the dataset!)

    1. **Only fine-tuning**: Don't have to do anything to download the dataset

    2. **Fine-tuning with Object Description**: Download 

        1. Red Box Detection Results(for regional task)

        2. Object Detection Results(for other tasks)
        
    
    3. **Fine-tuning with Segmented Tokens**: Download

        1. Red Box Detection Results(for regional task)

        2. Dino Segmented Results(for other tasks)

        3. YOLO Segmented Results(for other tasks)

### 1. Red Box Detection(For Region Task)

* Two options to download the dataset, conduct:

    1. If data exists:```bash scripts/preprocess_data_download/regional_cropped_images.sh```

    2. ```python modules/red_box_detection.py```

### 2. Object Detection(For Other Tasks)

* Two options to download the dataset, conduct:

    1. If data exists: ```bash scripts/preprocess_data_download/detection_objects.sh```

    2. ```python modules/detect_objects.py```

### 3. Segmented Images

* Two options to download the dataset, conduct:

    1. If data exists: ```bash scripts/preprocess_data_download/segment_objects.sh```

    2. ```python modules/segment_objects.py```

## E. Conduct Training

* There are some strategies you could choose to train:

    1. Strategy (1): Only fine-tuning

    2. Strategy (2): Add Object Detection Prompts 

    3. Strategy (3): Concat Object Segment Image's Tokens

### Strategy (1)

* Conduct 

    ```CUDA_VISIBLE_DEVICES=$1 bash scripts/strategy/llava-v1.5-7b-lora_5/finetune.sh```

    * ```$1```: GPU Number

### Strategy (2)

* Conduct 

    ```CUDA_VISIBLE_DEVICES=$1 bash scripts/strategy/llava-v1.5-7b-lora_add_obj_info_prompt_3/finetune.sh```

    * ```$1```: GPU Number

### Strategy (3)

* Conduct 

    ```CUDA_VISIBLE_DEVICES=$1 bash scripts/strategy/llava-v1.5-7b-lora_add_seg_img_token_5/finetune.sh```

    * ```$1```: GPU Number

# (Optional) Evaluation

* Two evaluation scripts to evaluate the performance of your model in **validation set**.

    1. ```Gemini evaluation```: this file is identical to the one we used in Codalab

        ```
        python3 evaluation/gemini_eval.py --prediction <you predicted json file> --api_key <your gemini api key>
        ```

        * Before evaluation, you have to install Gemini API. 

            ```
            pip install -q -U google-generativeai
            ```
        
        * Please refer to the following command. For more details, please refer to [Gemini API](https://ai.google.dev/gemini-api/docs/quickstart?hl=zh-tw&_gl=1*ciqklc*_up*MQ..&gclid=Cj0KCQiAgJa6BhCOARIsAMiL7V8rppSkxxeqt-eVsCczUZ8Iz2mXXiTi1EkuP7K2xalpBYOk9HLgbv0aAqAIEALw_wcB&lang=python).

        * Notes: Once you install ```google-generativeai```, if you want to go back to train, you have to conduct ```pip install protobuf==3.20``` to reverse the version of ```protobuf```

    2. ```Llama evaluation```: Since Gemini API has daily usage limits for free accounts, we provide a local testing option using LLaMA-3 as the LLM base model. Note that using llama_eval.py requires approximately 16GB of GPU memory.

        ```
        python3 evaluation/llama_eval.py --prediction <you predicted json file>
        ```

* For the argument `--prediction`, you should provide the json file which format is identical to "submission.json" described in [Submission Rules](#Submission-Rules).
* Both files will return the LLM judges and BLEU score of your predicted json file. The `Total score` is calculated by the following formula: `0.8 * LLM Score + 0.2 * BLEU-3`
    
    ```
    Genral score: x.xx
    Reasoning score: x.xx
    Suggestion score: x.xx
    LLM judges: x.xx
    Bleu_1 score: x.xx
    Bleu_2 score: x.xx
    Bleu_3 score: x.xx
    Bleu_4 score: x.xx
    Total score: x.xx
    ```
    
    `Notes:`
    * Since the total number of validation set is over the limit of free Gemini API, we suggest testing with only a small subset of the validation set when using Gemini API evaluation.
    * The results from LLaMA-3 may differ from Gemini's evaluation. Please use LLaMA-3's results **only as a reference**.
    * The supplementary materials of using Gemini API and huggingface tokens can be found in [slides](https://docs.google.com/presentation/d/1eeXx_dL0OgkDn9_lhXnimTHrE6OYvAiiVOBwo2CTVOQ/edit#slide=id.g31b10de1f8f_7_155).

# Results

|Strategy|Segmentation|Depth Map|RAG|Training Epochs|temperature|top_p|num_beams|Blue_3|General|Regional|Suggestion|LLM_Judge|Total Score|
|-----|-----|-------|----|----|------|---|---|---|---|---|---|---|---|
|Finetune LoRA only                                              |❌        |❌|❌|3|0.2|None|1|0.346|4.833|4.873|4.897|4.868|3.963|
|Finetune LoRA only                                              |❌        |❌|❌|5|0.2|None|1|0.339|5.693|4.843|4.450|4.996|4.064|
|Finetune LoRA only                                              |❌        |❌|❌|5|0  |0.9 |3|0.293|4.843|4.637|4.847|4.776|3.879|
|Finetune LoRA only                                              |❌        |❌|❌|6|0.2|None|1|0.337|4.753|4.660|4.917|4.777|3.889|
|Finetune LoRA only                                              |❌        |❌|✅|6|0.2|None|1|0.645|3.043|3.907|4.390|3.780|3.153|
|Finetune LoRA only("As a car...")                               |❌        |❌|❌|6|0.2|None|1|0.294|5.763|4.893|4.487|5.048|4.097|
|Add Seg. Token                                                  |✅(Token) |❌|❌|3|0.2|None|1|0.438|4.920|4.873|4.577|4.790|3.92|
|Add Seg. Token                                                  |✅(Token) |❌|❌|5|0.2|None|1|0.326|5.453|5.103|4.497|5.018|4.079|
|Add Seg. Prompt(old)                                            |✅(Prompt)|❌|❌|3|0.2|None|1|0.356|5.533|5.123|4.403|5.020|4.087|
|Add Seg. Prompt(old)                                            |✅(Prompt)|❌|❌|3|0  |0.9 |3|0.333|5.610|4.890|4.547|5.046|4.103|
|Add Seg. Prompt, Depth(old)                                     |✅(Prompt)|✅|❌|2|0.2|None|1|0.414|4.447|4.820|4.693|4.653|3.806|
|Add Seg. Prompt, Depth(new)("As a car...")                      |✅(Prompt)|✅|❌|3|0.2|None|1|0.357|5.230|5.210|4.357|4.932|4.017|
|Add Seg. Prompt, Depth(new)("As a car...") & Suggestion(Another)|✅(Prompt)|✅|❌|3|0.2|None|1|0.351|5.410|5.247|4.630|5.096|4.147|