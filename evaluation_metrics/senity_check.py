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
DATA_ROOT = r'/home/data/data/ILSVRC/Data/CLS-LOC/val_simple'      # Input your datasets link
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
pretrained_model = models.resnet50(pretrained=True).eval().to(DEVICE)
random_model = models.resnet50(pretrained=False).eval().to(DEVICE)

# Select interpretable methods
method_class = GradCAM

# The layers of the pre-trained model
target_layers_pretrained = [pretrained_model.layer4]

cam_pretrained = method_class(model=pretrained_model, target_layers=target_layers_pretrained)

# The layers of the un-trained model
target_layers_random = [random_model.layer4]

cam_random = method_class(model=random_model, target_layers=target_layers_random)


# Utility function: Determine whether it depends on the gradient
def requires_gradients(cam_instance):
    gradient_methods = (GradCAM, GradCAMElementWise, ShapleyCAM, FinerCAM, EigenGradCAM)
    return isinstance(cam_instance, gradient_methods)

# The main process of Senity Check
def evaluate_sanity(dataloader, cam_pretrained, cam_random):
    similarity_scores = []

    grad_required = requires_gradients(cam_pretrained)

    for images, _ in tqdm(dataloader, desc="Evaluating Explanation Sanity"):
        images = images.to(DEVICE)

        # 1. Get the Top-1 category while allow no_grad
        with torch.no_grad():
            logits = pretrained_model(images)
            probs = torch.softmax(logits, dim=1)
            top_classes = torch.argmax(probs, dim=1)

        targets = [ClassifierOutputTarget(c.item()) for c in top_classes]

        # 2. Explanation of pre-trained models
        if requires_gradients(cam_pretrained):
            pretrained_model.zero_grad()
            if hasattr(cam_pretrained, "_ensure_hooks_exist"):
                cam_pretrained._ensure_hooks_exist()
            cam_maps_pretrained = cam_pretrained(input_tensor=images, targets=targets)
        else:
            with torch.no_grad():
                cam_maps_pretrained = cam_pretrained(input_tensor=images, targets=targets)

        # 3. Random initialization model explanation
        if requires_gradients(cam_random):
            random_model.zero_grad()
            if hasattr(cam_random, "_ensure_hooks_exist"):
                cam_random._ensure_hooks_exist()
            cam_maps_random = cam_random(input_tensor=images, targets=targets)
        else:
            with torch.no_grad():
                cam_maps_random = cam_random(input_tensor=images, targets=targets)

        # 4. Calculate similarity
        cam_maps_pretrained = torch.from_numpy(cam_maps_pretrained).flatten(start_dim=1)
        cam_maps_random = torch.from_numpy(cam_maps_random).flatten(start_dim=1)

        for map_pre, map_rand in zip(cam_maps_pretrained, cam_maps_random):
            map_pre = map_pre.cpu().numpy()
            map_rand = map_rand.cpu().numpy()

            if np.std(map_pre) < 1e-5 or np.std(map_rand) < 1e-5:
                continue

            r, _ = pearsonr(map_pre, map_rand)
            similarity_scores.append(r)

        torch.cuda.empty_cache()

    similarity_scores = [s for s in similarity_scores if not np.isnan(s)]
    avg_similarity = np.mean(similarity_scores)
    return avg_similarity

# main process
if __name__ == '__main__':
    try:
        avg_similarity = evaluate_sanity(dataloader, cam_pretrained, cam_random)
        print(f"\n🧩 Explanation Sanity Check (Pearson similarity): {avg_similarity:.4f}")
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise
