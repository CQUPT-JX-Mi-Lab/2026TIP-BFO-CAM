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
from tqdm import tqdm
from scipy.stats import pearsonr

# Configuration parameter
DATA_ROOT = r'/home/data/data/ILSVRC/Data/CLS-LOC/val_simple'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32

# Image preprocessing
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

# Loading model
model = models.resnet50(pretrained=True)
model.eval().to(DEVICE)

target_layers = [model.layer4]
cam = GradCAM(model=model, target_layers=target_layers)


# Category sensitivity assessment function
def evaluate_class_sensitivity(dataloader, model, cam):
    cs_scores = []
    for images, _ in tqdm(dataloader, desc="Evaluating Class Sensitivity"):
        images = images.to(DEVICE)

        with torch.no_grad():
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            top_classes = torch.argmax(probs, dim=1)
            bottom_classes = torch.argmin(probs, dim=1)

        # Generate saliency maps of the max/min category for each image respectively
        targets_max = [ClassifierOutputTarget(c.item()) for c in top_classes]
        targets_min = [ClassifierOutputTarget(c.item()) for c in bottom_classes]

        cam_max = cam(input_tensor=images, targets=targets_max)  # (B, H, W)
        cam_min = cam(input_tensor=images, targets=targets_min)

        cam_max = torch.from_numpy(cam_max).flatten(start_dim=1)
        cam_min = torch.from_numpy(cam_min).flatten(start_dim=1)

        for max_map, min_map in zip(cam_max, cam_min):
            max_map = max_map.cpu().numpy()
            min_map = min_map.cpu().numpy()
            if np.std(max_map) < 1e-5 or np.std(min_map) < 1e-5:
                continue
            r, _ = pearsonr(max_map, min_map)
            cs_scores.append(r)

        torch.cuda.empty_cache()

    cs_scores = [s for s in cs_scores if not np.isnan(s)]
    avg_cs = np.mean(cs_scores)
    return avg_cs

# main process
if __name__ == '__main__':
    try:
        avg_cs, sensitivity_score = evaluate_class_sensitivity(dataloader, model, cam)
        print(f"\n📊 Class Sensitivity (Pearson correlation): {avg_cs:.4f}")
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise
