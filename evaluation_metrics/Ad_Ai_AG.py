import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import (
    GradCAM, ScoreCAM,
    DFOCAM, KPCA_CAM, ShapleyCAM,
    FinerCAM
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Configuration parameter
DATA_ROOT = r'/home/data/data/ILSVRC/Data/CLS-LOC/val_simple'  # ImageNet验证集路径
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
CAM_THRESHOLD = 0.5  # The threshold for binarizing mask

# 1. Data loading and preprocessing
class ImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    samples.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = ImageNetDataset(DATA_ROOT, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# 2. Loading model and methods
model = models.resnet50(pretrained=True)
model.eval()
model.to(DEVICE)

# Ensure that all parameters of the model allow for gradient calculation
for param in model.parameters():
    param.requires_grad = True

target_layers = [model.layer4]

cam = GradCAMElementWise(model=model, target_layers=target_layers)

# 3. AD/AI evaluation function
def evaluate_ad_ai_ag(dataloader, model, cam):
    total_ad = 0.0
    total_ai = 0.0
    total_ag = 0.0
    total_samples = 0


    for images, labels in tqdm(dataloader, desc="Evaluating AD/AI"):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.no_grad():
            logits = model(images)
            y_orig = torch.softmax(logits, dim=1).gather(1, labels.unsqueeze(1)).squeeze()

        # Get CAM
        try:
            targets = [ClassifierOutputTarget(label.item()) for label in labels]
            grayscale_cam = cam(input_tensor=images, targets=targets)

            # Convert to a Tensor and interpolate to the image size
            grayscale_cam = torch.from_numpy(grayscale_cam).to(DEVICE)

            # Retain the significant areas (the rest are 0)
            with torch.no_grad():

                # Binarized saliency map
                binary_mask = (grayscale_cam > CAM_THRESHOLD).float()

                kept_images = images * binary_mask.unsqueeze(1)
                y_keep = torch.softmax(model(kept_images), dim=1).gather(1, labels.unsqueeze(1)).squeeze()

                # Average Drop
                ad = torch.relu(y_orig - y_keep) / (y_orig + 1e-8) * 100  # 百分比形式
                total_ad += ad.sum().item()

                # Average Increase
                ai = (y_keep > y_orig).float()
                total_ai += ai.sum().item()

                # Average Gain
                gain = torch.relu(y_keep - y_orig) / (1 - y_orig + 1e-8) * 100
                total_ag += gain.sum().item()

            total_samples += images.size(0)

        finally:
            torch.cuda.empty_cache()

    avg_ad = total_ad / total_samples
    avg_ai = total_ai / total_samples
    avg_ag = total_ag / total_samples
    return avg_ad, avg_ai, avg_ag



# 4. 评估函数
if __name__ == '__main__':
    try:
        avg_ad, avg_ai, avg_ag = evaluate_ad_ai_ag(dataloader, model, cam)
        print(f"\nEvaluation Results:")
        print(f"Average Drop (AD): {avg_ad:.2f}%")
        print(f"Average Increase (AI): {avg_ai * 100:.2f}%")
        print(f"Average Gain (AG): {avg_ag:.2f}%")
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise
