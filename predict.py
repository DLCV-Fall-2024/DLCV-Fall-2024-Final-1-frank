import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from collections import defaultdict
import re
import cv2
import time

from LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from LLaVA.llava.conversation import conv_templates, SeparatorStyle
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
import numpy as np

from segmentation.DINO_seg import load_groundingdino_model, get_bounding_boxes, process_but_no_save, load_sam_model
from segmentation.crop_regional import crop
from segmentation.YOLO_seg import process_but_no_save_segmentation, load_detection_model
from depth_map.DINO_with_labels import load_depth_anything_model, process_but_no_save_results


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    with open(args.question_file, 'r') as f:
        questions_test = json.load(f)
    questions = get_chunk(questions_test, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    # ans_file = open(answers_file, "w")
    
    # load for the segmentation module if needed
    if args.add_region_prompt or args.add_region_token or args.add_region_depth_prompt:
        processor, groundingdino_model, device = load_groundingdino_model()
    if args.add_region_token:
        sam_predictor = load_sam_model()
        YOLO_detector = load_detection_model()
    
    results = {}
    for line in tqdm(questions):
        idx = line["id"]
        image_file = f"{line['id']}.png"
        qs = line["conversations"][0]["value"].replace("<image>", "")
        
        # get the image and the image weight and height
        img_path = os.path.join(args.image_folder, image_file)
        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        
        regional_img = None 
        
        # for add region_prompt
        if args.add_region_prompt:
            appending_prompt = ''
            if  "regional" in idx:
                crop_result = crop(image_path=img_path)
                if crop_result is not None:
                    coordination = crop_result[1]
                    coordination = [( round(ele/image.width, 4) if i%2 == 0 else round(ele/image.height, 4)) for i, ele in enumerate(coordination)]
                    assert min(coordination) >=0 and max(coordination) <= 1
                    appending_prompt = f'''You have to first examine whether there is a red bounding box (x_min, y_min, x_max, y_max): {coordination} \n If it is, just stay focus on the object inside the bounding box'''
            else:
                boxes, _, labels = get_bounding_boxes(np.array(image), processor, groundingdino_model, device)
                if len(boxes) == 0:
                    continue
                result_dict = defaultdict(list)
                for bbox, label in zip(boxes, labels):
                    result_dict[label].append([( round(ele/image.width, 4) if i%2 == 0 else round(ele/image.height, 4)) for i, ele in enumerate(bbox.astype(int).tolist())])
                result_dict = dict(result_dict)
                result_text = ""
                for key in result_dict:
                    result_text += f"{key}: {result_dict[key]} \n"
                appending_prompt = f'''You can refer some important objects given the following information(The format would be <object>:[<x_min, y_min, x_max, y_max>, ...]: \n"{result_text}'''
            qs += appending_prompt
            torch.cuda.empty_cache()
        if args.add_region_token:
            if "regional" in idx:
                crop_result = crop(img_path)
                if crop_result:
                    regional_img = crop_result[0]
                else:
                    cv2_image = cv2.imread(img_path)
                    hsv = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2HSV)

                    lower_red1 = np.array([0, 100, 100])   # 紅色下界1
                    upper_red1 = np.array([10, 255, 255]) # 紅色上界1
                    lower_red2 = np.array([160, 100, 100]) # 紅色下界2
                    upper_red2 = np.array([180, 255, 255]) # 紅色上界2

                    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                    mask = cv2.bitwise_or(mask1, mask2)

                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        
                        mask_image = np.zeros_like(cv2_image)
                        mask_image[y:y+h, x:x+w] = cv2_image[y:y+h, x:x+w]
                        regional_img = Image.fromarray(mask_image)
                qs = re.sub(r"(describe the object)", r"\1 <region>", qs)
            else:
                regional_img = process_but_no_save(image, idx, processor=processor, groundingdino_model=groundingdino_model, device=device, sam_predictor=sam_predictor)
                if not regional_img:
                    regional_img = process_but_no_save_segmentation(image=image, sam_predictor=sam_predictor, detection_model=YOLO_detector)
                qs = re.sub(r"(Focus on objects)", r"\1 <region>", qs) 
        if args.add_region_depth_prompt:
            appending_prompt = ''
            depth_model = load_depth_anything_model(device)
            object_infos = process_but_no_save_results(image, idx, processor, groundingdino_model, depth_model, device)
            if object_infos is not None:
                if "regional" not in idx:
                    appending_prompt = "\nHere are some important objects you should focus on:\n"
                    for object_info in object_infos:
                        classification, box, depth = object_info["class"], object_info["box"], object_info["depth_category"]["depth_category"] 
                        box = [(round(ele/image.width, 4) if i%2 == 0 else round(ele/image.height, 4)) for i, ele in enumerate(box)]
                        appending_prompt += f'''Object: {classification} \n Box (x_min, y_min, x_max, y_max): {','.join([str(ele) for ele in box])} \n Distance to our eco car: {depth}'''
                else:
                    crop_result = crop(img_path)
                    if crop_result:
                        single_box = crop_result[1]
                        for object_info in object_infos:
                            classification, box, depth = object_info["class"], object_info["box"], object_info["depth_category"]["depth_category"] 
                            if not ((box[2] < single_box[0]) or (box[0] > single_box[2]) or (box[3] < single_box[1]) or (box[1] > single_box[3])):
                                box = [(round(ele/image.width, 4) if i%2 == 0 else round(ele/image.height, 4)) for i, ele in enumerate(box)]
                                appending_prompt += f'''Object: {classification} \n Box (x_min, y_min, x_max, y_max): {','.join([str(ele) for ele in box])} \n Distance to our eco car: {depth}'''
                        appending_prompt = "\nHere is the important object you should focus on:\n" + appending_prompt if appending_prompt else ""
            else:
                crop_result = crop(img_path)
                if crop_result:
                    box = crop_result[1]
                    box = [(round(ele/image.width, 4) if i%2 == 0 else round(ele/image.height, 4)) for i, ele in enumerate(box)]
                    appending_prompt = f'''You should stay focus on the object(x_min, y_min, x_max, y_max): {','.join(str(ele) for ele in box)}'''
            qs += appending_prompt
                
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt', add_region_token=args.add_region_token, add_detection_token=args.add_detection_token).unsqueeze(0).cuda()

        image_tensor = process_images([image], image_processor, model.config)[0]
        if regional_img: 
            regional_tensor = process_images([regional_img], image_processor, model.config)[0].unsqueeze(0).half().cuda()
        else:
            regional_tensor = None 

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                regional_images=regional_tensor,
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                add_detection_token = args.add_detection_token,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # ans_id = shortuuid.uuid()
        # ans_file.write(json.dumps({"question_id": idx,
        #                            "prompt": cur_prompt,
        #                            "text": outputs,
        #                            "answer_id": ans_id,
        #                            "model_id": model_name,
        #                            "metadata": {}}) + "\n")
        # ans_file.flush()
        # print(qs)
        # print(idx, outputs)
        
        results[idx] = outputs
        # print(outputs)
    # ans_file.close()
    with open(args.answers_file, 'w') as f:
        json.dump(results, f, indent=4) 
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder",  type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=3)
    
    parser.add_argument('--add_region_token', action='store_true')
    parser.add_argument("--add_region_prompt", action='store_true')
    parser.add_argument("--add_region_depth_prompt", action='store_true')
    parser.add_argument('--add_detection_token', action='store_true')
    args = parser.parse_args()

    eval_model(args)
