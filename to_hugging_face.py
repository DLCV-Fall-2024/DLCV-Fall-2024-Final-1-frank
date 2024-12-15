from huggingface_hub import HfApi
import os
from pathlib import Path

def upload_local_dataset_to_huggingface(local_directory, repo_name, organization=None):
    """
    Args:
    - local_directory
    - repo_name
    """
    api = HfApi()
    
    if not os.path.exists(local_directory):
        raise ValueError(f"directory {local_directory} does not exist")
    
    if organization:
        repo_full_name = f"{organization}/{repo_name}"
        api.create_repo(
            repo_id=repo_full_name, 
            repo_type="dataset",
            exist_ok=True  
        )
    else:
        repo_full_name = repo_name
        api.create_repo(
            repo_id=repo_name, 
            repo_type="dataset",
            exist_ok=True
        )
    
    for root, dirs, files in os.walk(local_directory):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_directory)
            
            try:
                api.upload_file(
                    path_or_fileobj=local_file_path,
                    path_in_repo=relative_path,
                    repo_id=repo_full_name,
                    repo_type="dataset"
                )
                print(f"Successfully uploads: {relative_path}")
            except Exception as e:
                print(f"Having the error: {e} during upload of {relative_path}.")

def main():
    local_directory = "./YOLO_SAM_segmentation_results"  
    repo_name = "pianoholic/YOLO_SAM_segmentation_results" 
    organization = None  

    upload_local_dataset_to_huggingface(local_directory, repo_name, organization)

if __name__ == "__main__":
    main()