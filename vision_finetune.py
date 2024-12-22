import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import functional as F
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ColorJitter, RandomHorizontalFlip, RandomRotation
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import random


# ------------------------
# (A) Dataset 
# ------------------------
class COCODataset(Dataset):
    def __init__(self, img_dir, ann_path, categories_json, transforms=None):
        self.img_dir = img_dir
        self.ann_path = ann_path
        self.transforms = transforms

        with open(self.ann_path, 'r') as f:
            coco_json = json.load(f)

        self.images = coco_json["images"]
        self.annotations = coco_json["annotations"]
        self.categories = coco_json["categories"]

        self.image_id_to_info = {img["id"]: img for img in self.images}
        self.image_id_to_anns = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.image_id_to_anns:
                self.image_id_to_anns[img_id] = []
            self.image_id_to_anns[img_id].append(ann)

        with open(categories_json, 'r') as f:
            category_data = json.load(f)
        self.id_to_name = {cat["id"]: cat["name"] for cat in category_data["categories"]}
        self.name_to_id = {name: i for i, name in enumerate(self.id_to_name.values())}

        self.ids = [img["id"] for img in self.images]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img_info = self.image_id_to_info[image_id]
        file_name = img_info["file_name"]

        img_path = os.path.join(self.img_dir, file_name)
        img = Image.open(img_path).convert("RGB")

        ann_list = self.image_id_to_anns.get(image_id, [])

        if len(ann_list) > 0:
            ann = random.choice(ann_list)
            x, y, w, h = ann["bbox"]
            label_id = self.name_to_id[self.id_to_name[ann["category_id"]]]
            cropped_img = img.crop((x, y, x + w, y + h))  

            if self.transforms:
                cropped_img = self.transforms(cropped_img)

            return cropped_img, torch.tensor(label_id)
        else:
            if self.transforms:
                img = self.transforms(img)
            return img, torch.tensor(self.name_to_id["background"])


# ------------------------
# (B) Transform 
# ------------------------
def get_transforms():
    return Compose([
        Resize((224, 224)),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=10),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


# ------------------------
# (C) ViT 
# ------------------------
def get_vit_model(num_classes):
    model = ViTForImageClassification.from_pretrained("google/vit-large-patch16-224-in21k")
    model.classifier = torch.nn.Linear(model.config.hidden_size, num_classes)
    return model


# ------------------------
# (D) Train
# ------------------------
def train_one_epoch(model, optimizer, data_loader, device, epoch, grad_accum_steps=4):
    model.train()
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    optimizer.zero_grad()

    with tqdm(data_loader, desc=f"Epoch {epoch+1}", unit="batch") as pbar:
        for i, (imgs, labels) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs).logits
            loss = criterion(outputs, labels)
            loss.backward()

            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(data_loader):
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

    return total_loss / len(data_loader)


# ------------------------
# (E) Main
# ------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    categories_json_path = "categories.json"

    dataset_1 = COCODataset(
        img_dir="./images_1",
        ann_path="./annotations_1.json",
        categories_json=categories_json_path,
        transforms=get_transforms()
    )
    dataset_2 = COCODataset(
        img_dir="./images_2",
        ann_path="./annotations_2.json",
        categories_json=categories_json_path,
        transforms=get_transforms()
    )

    dataset_train = torch.utils.data.ConcatDataset([dataset_1, dataset_2])

    train_size = int(0.8 * len(dataset_train))
    val_size = len(dataset_train) - train_size
    train_dataset, val_dataset = random_split(dataset_train, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=8
    )

    num_classes = len(dataset_1.name_to_id)

    model = get_vit_model(num_classes=num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    num_epochs = 20
    best_loss = float("inf")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        scheduler.step()

        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}")

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), "vit_finetuned.pth")
            print("Model saved to vit_finetuned.pth")