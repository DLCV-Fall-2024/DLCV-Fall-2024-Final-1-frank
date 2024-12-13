import os
import json
import argparse
import random
import string
import numpy as np
from collections import defaultdict

import torch
import transformers
from datasets import load_dataset
from scorer import Scorer

def formulate_template(few_shot_path, message, sample_type="general"):
    # system prompt
    messages = [{
        "role": "system", 
        "content": "You are an impartial judge tasked with evaluating the quality of predicted text provided by autonomous driving AI assistant. You will compare this prediction text to a reference text, focusing on the description of objects that influence the driving behavior of ego car, and the explanation of why these objects impact. Your evaluation criteria should include accuracy(checking if the predicted text correctly identifies objects mentioned the reference text), suppression hallucination(ensuring that objects not mentioned in the reference text are not erroneously included in the predicted text), correlation(sessing if the reasons for the objects' impact on the ego car's driving behavior are consistent between the reference and predicted text). Be as objective as possible. Do not allow the length of the predicted text to influence your evaluation. Maximize your text comprehension capabilities to freely match objects with high similarity, appropriately ignoring the relative positions and color attributes of the objects. After providing your short explanation, you must rate the response on a scale from 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[10]]\"."
    }]
    # add few-shot example
    template = "[The Start of Reference Text]\n{}\n[The End of Reference Text]\n\n[The Start of Prediction Text]\n{}\n[The End of Prediction Text]"
    if sample_type == "general" or sample_type == "suggestion":
        
        file = "scene_few_shot" if sample_type == "general" else "suggestion_few_shot"
        json_folder = os.path.join(few_shot_path, file)
        for sample in os.listdir(json_folder):
            json_path = os.path.join(json_folder, sample)
            data = json.load(open(json_path, "r"))
            # template = "[The Start of Reference Text]\n{}\n[The End of Reference Text]\n\n[The Start of Prediction Text]\n{}\n[The End of Prediction Text]"
            messages.append({
                "role": "user", 
                "content": template.format(data["reference"], data["prediction"])
            })
            messages.append({
                "role": "assistant", 
                "content": data["response"]
            })
    # add query
    messages.append({
        "role": "user", 
        "content": template.format(message["reference"], message["prediction"])
    })

    return messages

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--few_shot", type=str, default="few_shot")
    parser.add_argument("--prediction", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="ntudlcv/dlcv_2024_final1")
    parser.add_argument("--split", type=str, default="val")
    # parser.add_argument("--save", type=str, default="llama3_eval.json")
    parser.add_argument("--max_output_tokens", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":

    args = arguments()
    # Load model
    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="cuda",
        temperature = args.temperature,
    )
    # Load data
    reference = load_dataset(args.dataset_name, split=args.split, streaming=True)
    prediction = json.load(open(args.prediction, "r"))
    NLP_HYPOTHESIS = {key: [value.strip()] for key, value in prediction.items()}
    NLP_REFERENCE = {}

    result = defaultdict(list)
    # save = {}
    # fail_cases = []
    for data in reference:
        sample_id = data["id"]
        score = 0
        
        if sample_id not in prediction:
            continue

        message = {"reference": "", "prediction": prediction[sample_id]}
        message["reference"] = data["conversations"][1]["value"]
        sample_type = (sample_id.split("_")[1]).lower()
        messages = formulate_template(args.few_shot, message, sample_type)
        for max_tokens in [args.max_output_tokens, 1024]:
            try:
                outputs = pipeline(
                    messages,
                    max_new_tokens=max_tokens,
                )
                score = int(outputs[0]["generated_text"][-1]["content"].split('[[')[-1].split(']')[0])
                break
            except:
                continue

        # if score == 0:
        #     print(f"Missing score for sample: {sample_id}")
        #     fail_cases.append(sample_id)
        result[sample_type].append(score)

        # save[sample_id] = {
        #     "prediction": outputs[0]["generated_text"][-1]["content"],
        #     "score": score
        # }
        # with open(args.save, "w") as f:
        #     json.dump(save, f, indent=4)
        NLP_REFERENCE[sample_id] = [data["conversations"][1]["value"]]
    
    coco_eval = Scorer(NLP_HYPOTHESIS, NLP_REFERENCE)
    total_scores = coco_eval.evaluate()

    print("\n\n")
    # LLM judge
    total = []
    for sample_type, scores in result.items():
        score = np.mean(scores)
        print(f"{sample_type.capitalize()} score: {score:.3f}")
        total.append(score)
    print(f"LLM judges: {np.mean(total):.3f}")
    # print("Number of fail cases: ", len(fail_cases))

    # NLP metric
    for key, value in total_scores.items():
        print(f'{key}: {value:.3f}')

    total_score = np.mean(total) * 0.8 + total_scores["Bleu_3"] * 0.2
    print(f"Total score: {total_score:.3f}") 