from datasets import load_dataset
import argparse, json, os
from PIL import Image
from tqdm import tqdm

MAX_NUM = 6884

def _parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dataset_directory', type=str, help='The folder for storing the datasets', default="dataset")
    parser.add_argument('--dataset_types', type=str, nargs='+', help="the purpose list for the datasets", default=["train", "val", "test"])
    parser.add_argument('--max_dataset_num', type=int, help="Since the total dataset is too large, so we instead search for the first several data. \
        You could still download the whole dataset if set the value to be -1", default=10)
    return parser.parse_args()

def main():
    args = _parser_args()
    
    # # check the column/key names for the dataset
    # # Column names: id, image, conversation
    # check_dataset = list(load_dataset("ntudlcv/dlcv_2024_final1", split="train", streaming=True).take(1))
    # key_list = check_dataset[0].keys()
    # key_info = [f"{key}: {type(check_dataset[0][key])}\n" for key in key_list]
    # print(f"The columns(keys) in the dataset are:\n{''.join(key_info)}")
    
    for dataset_type in tqdm(args.dataset_types):
        # initialize
        output_dir = os.path.join(args.output_dataset_directory, dataset_type)
        os.makedirs(output_dir, exist_ok=True)
        annotation = []
        
        # load dataset, but only load some data
        # streaming = True: would not load the whole dataset
        dataset = load_dataset("ntudlcv/dlcv_2024_final1", split=dataset_type, streaming=True)
        
        # load the subset of the dataset
        dataset_iter = iter(dataset)
        args.max_dataset_num = args.max_dataset_num if args.max_dataset_num != -1 else MAX_NUM
        for _ in range(args.max_dataset_num):
            try:
                item = next(dataset_iter)
                id_, conversation, image = item["id"], item["conversations"], item["image"]
                # insert annotation
                data = {"id":id_, "conversations":conversation}
                annotation.append(data)
                
                # save image
                img_dir = os.path.join(output_dir, "images")
                os.makedirs(img_dir, exist_ok=True)
                img_path = os.path.join(img_dir, f"{id_}.png")
                image.save(img_path)
            except StopIteration:
                print("End of dataset")
                break
        # save annotation
        annotation_path = os.path.join(output_dir, "annotation.json")
        with open(annotation_path, 'w') as f:
            json.dump(annotation, f, indent=4)
            
if __name__ == "__main__":
    main()