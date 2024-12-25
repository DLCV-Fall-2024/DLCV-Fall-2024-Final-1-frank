import argparse
import torch
import os
import json
from tqdm import tqdm
from collections import defaultdict
import re
import cv2
import json


from LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_SEG_IMAGE_TOKEN
from LLaVA.llava.conversation import conv_templates, SeparatorStyle
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
import numpy as np

from modules.segment_objects import SegDino, SegYOLO
from modules.detect_objects import DetectObjectModel
from modules.red_box_detection import CropRedBoxModel

from dotenv import load_dotenv
from huggingface_hub import login
load_dotenv()
token = os.getenv("HUGGINGFACE_HUB_TOKEN")
login(token)

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
    
    if args.suggestion_model_path:
        assert args.add_obj_info_prompt and not args.add_seg_img_token
        suggestion_model_path = os.path.expanduser(args.suggestion_model_path)
        suggestion_model_name = get_model_name_from_path(suggestion_model_path)
        suggestion_tokenizer, suggestion_model, suggestion_image_processor, suggestion_context_len = load_pretrained_model(suggestion_model_path, args.model_base, suggestion_model_name)

    with open(args.question_file, 'r') as f:
        questions_test = json.load(f)
    questions = get_chunk(questions_test, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    results = {}
    
    # load for the image
    det_obj_model = DetectObjectModel()
    if args.add_obj_info_prompt:
        red_box_crop_model = CropRedBoxModel()
        
    if args.add_seg_img_token:
        red_box_crop_model = CropRedBoxModel()
        seg_dino = SegDino()
        seg_yolo = SegYOLO()

    for line in tqdm(questions):
        idx = line["id"]
        image_file = f"{line['id']}.png"
        
        # get the image and the image weight and height
        img_path = os.path.join(args.image_folder, image_file)
        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        
        qs = line["conversations"][0]["value"].replace("<image>", "")
        qs = "\n\nYou are an experienced car driver, \
            enable to point out all details need to be focused while driving. \
                An image from the driver's seat of a ego car is given, \
                    corresponded to a question. You need to answer the question with your analysis from image.\n" + f" Question\n\"{qs} \n"
        
        seg_image = None 
        
        # for add region_prompt
        if args.add_obj_info_prompt:
            appending_prompt = ''
            if  "regional" in idx:
                red_box_info = red_box_crop_model.get_red_box(image=image, image_id=idx)
                if red_box_info is not None:
                    red_box_info = red_box_info["detection_information"]
                    coordinate = [( round(ele/image.width, 4) if i%2 == 0 else round(ele/image.height, 4)) for i, ele in enumerate(red_box_info["box"])]
                    depth = red_box_info["depth_category"]
                    label = red_box_info["predicted_label"]
                    appending_prompt = f'\n To deal with it easily, you only need to focus on the object: \n * object: {label} \n * coordinate: {coordinate} \n * distance: {depth} \n'
            elif "suggestion" in idx and args.suggestion_model_path:
                pass
            else:
                object_infos = det_obj_model.get_objs_info(image, idx)
                if object_infos:
                    appending_prompt = f'\n You only need to focus on the objects: \n'
                    for object_info in object_infos:
                        coordinate = [( round(ele/image.width, 4) if i%2 == 0 else round(ele/image.height, 4)) for i, ele in enumerate(object_info["box"])]
                        depth = object_info["depth_category"]["depth_category"]
                        label = object_info["class"]
                        appending_prompt += f"\n * Object: {label} \n * Coordinate(x_min, y_min, x_max, y_max):{coordinate}. \n * Distance: {depth} \n"
            qs += appending_prompt
            torch.cuda.empty_cache()
        
        # for add image seg token
        if args.add_seg_img_token:
            if "regional" in idx:
                crop_result = red_box_crop_model.get_red_box(image=image, image_id=idx)
                if crop_result:
                    seg_image = crop_result["segmented_image"]
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
                        seg_image = Image.fromarray(mask_image)
                qs = re.sub(r"(describe the object)", rf"\1 {DEFAULT_SEG_IMAGE_TOKEN}", qs)
            else:
                seg_image = seg_dino.get_seg_image(image=image, image_id=idx)
                if not seg_image:
                    seg_image = seg_yolo.get_seg_image(image=image, image_id=idx)
                qs = re.sub(r"(Focus on objects)", rf"\1 {DEFAULT_SEG_IMAGE_TOKEN}", qs) 
        # if args.add_detection_token:
        #     qs += "The feature of the labels and bounding boxes are in <detection>."
        
        def generate(_tokenizer, _model, _image_processor, qs=qs, args=args, seg_image=seg_image):
            if _model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            qs += "\"\n Your answer:"
            
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, _tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt', add_seg_img_token=args.add_seg_img_token).unsqueeze(0).cuda()

            image_tensor = process_images([image], _image_processor, _model.config)[0]
            if seg_image: 
                regional_tensor = process_images([seg_image], _image_processor, _model.config)[0].unsqueeze(0).half().cuda()
            else:
                regional_tensor = None 

            with torch.inference_mode():
                output_ids = _model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    seg_images=regional_tensor,
                    image_sizes=[image.size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # add_detection_token = args.add_detection_token,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            outputs = _tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            return outputs
        
        if "suggestion" in idx and args.suggestion_model_path:
            outputs = generate(_tokenizer=suggestion_tokenizer, _model=suggestion_model, _image_processor=suggestion_image_processor)
        else:
            outputs = generate(_tokenizer=tokenizer, _model=model, _image_processor=image_processor)
        
        results[idx] = outputs
        # print(idx)
        # print(qs)
        # print(outputs)
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
    
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    
    parser.add_argument('--add_seg_img_token', action='store_true')
    parser.add_argument("--add_obj_info_prompt", action='store_true')
    parser.add_argument('--suggestion_model_path', default=None, type=str)
    
    args = parser.parse_args()

    eval_model(args)
