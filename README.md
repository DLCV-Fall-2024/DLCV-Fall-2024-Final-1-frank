# Notes

* All executions are performed in the **root** directory.

# Folder Description

* ```few_shot```: The example of few-shot

* ```images```: The image used in ```README_original.md```

# Initialization

# Execution 

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

# Results

# Supplement

## Preview Dataset 

* Conduct the following code

    ```
    python3 supplement/data_download.py
    ```
