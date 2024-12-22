# Notes

* All executions are performed in the **root** directory.

# Folder Description

* ```reference```: Store the previous README data

* ```evaluation```: Code for evaluating the results

* ```supplement```: Code for other purpose

* ```LLaVA```: Code for LLaVA repository

# Initialization

## Environment Setup

1. Create a new environment (python version needs to `>=3.10`)
    
    ```
    conda create -n <your_env_name> python=3.10 -y
    conda activate <your_env_name>
    pip install -r requirement.txt
    ```

2. Install Gemini API: To install Gemini API, please refer to the following command. For more details, please refer to [Gemini API](https://ai.google.dev/gemini-api/docs/quickstart?hl=zh-tw&_gl=1*ciqklc*_up*MQ..&gclid=Cj0KCQiAgJa6BhCOARIsAMiL7V8rppSkxxeqt-eVsCczUZ8Iz2mXXiTi1EkuP7K2xalpBYOk9HLgbv0aAqAIEALw_wcB&lang=python).
    
    ```
    pip install -q -U google-generativeai
    ```

    * Notes: Once you install ```google-generativeai```, if you want to go back to train, you have to conduct ```pip install protobuf==3.20``` to reverse the version of ```protobuf```

3. Download the weights of pre-trained model:

    ```
    bash pretrained_download.sh
    ```

4. (For training) You have to create a ```wandb``` account to trace the training result [here](https://wandb.ai/)


## Data Preparation

* Conduct the following instruction

    ```
    python3 data_download.py --dataset_types train val test
    ```

    * The data would be downloaded in ```data``` folder

    * You could modify given parameters in ```--dataset_types``` to download which split of data you want to download. e.g. ```--dataset_types train``` for only download the training dataset

    * You could also choose to add ```--max_dataset_num``` to download partial of the dataset

* **Watch Out!!**: Download the whole ```train``` dataset might requires 4-5 hours

# Execution 

## Training

* Conduct the following scripts 

    ```
    bash finetune.sh
    ```

    * If your environment has more than one gpu, you need to choose one gpu to conduct, which you needs to add ```CUDA_VISIBLE_DEVICES = <your gpu num>``` to assign which gpu you are going to use.

## Prediction

* After finetuning(i.e. conduct ```bash finetune.sh```), you can execute ```bash predict.sh```

    * You have to replace the parameter ```--model-path``` with what you given ```--output_dir``` when you train.

    * You could add ```CUDA_VISIBLE_DEVICES``` to assign which gpu you are going to use

## Evaluation

* Two evaluation scripts to evaluate the performance of your model in **validation set**.

    1. ```Gemini evaluation```: this file is identical to the one we used in Codalab

        ```
        python3 evaluation/gemini_eval.py --prediction <you predicted json file> --api_key <your gemini api key>
        ```

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

## Supplement: Segmentation 

* 


# Results

|Strategy|Segmentation|Depth Map|Training Epochs|Blue_3|General|Regional|Suggestion|LLM_Judge|Total Score|
|-----|-----|-------|-----|------|---|---|---|---|---|
|Finetune LoRA only|❌|❌|3|0.346|4.833|4.873|4.897|4.868|3.963|
|Finetune LoRA only|❌|❌|5|0.339|5.693|4.843|4.450|4.996|4.064|
|Add Seg. Token   |✅(Token)|❌|3|0.438|4.920|4.873|4.577|4.790|3.92|
|Add Seg. Prompt  |✅(Prompt)|❌|3|0.356|5.533|5.123|4.403|5.020|4.087|


# Supplement

## Preview Dataset 

* Conduct the following code

    ```
    python3 supplement/data_download.py
    ```
