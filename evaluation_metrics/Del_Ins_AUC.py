import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms, models
from pytorch_grad_cam import (
    GradCAM, ScoreCAM,
    DFOCAM, KPCA_CAM, ShapleyCAM,
    FinerCAM
)
from PIL import Image
from tqdm import tqdm

import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image, ImageFilter

# Configuration parameter
class Config:
    num_steps = 40
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    input_size = (224, 224)          # (W, H)

# Image preprocessing
def preprocess(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(Config.input_size, Image.BILINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.mean, std=Config.std)
    ])
    tensor = transform(image).unsqueeze(0).to(Config.device)
    return tensor, np.array(image) / 255.0        # (tensor, rgb[0-1])

# Evaluate a single image
def evaluate_single_image(model, cam_extractor, image_path,
                          mode='insertion', save_dir=None, verbose=False):
    # 1) Read the graph & predict the category
    input_tensor, rgb_image = preprocess(image_path)
    with torch.no_grad():
        logits = model(input_tensor)
    pred_class = logits.argmax(dim=1).item()

    # 2) Generate CAM and normalize it to 0 to 1
    cam = cam_extractor(input_tensor=input_tensor, targets=None)[0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # 3) (Optional) Visualization CAM
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = (1 - heatmap / 255.0) * 0.5 + rgb_image * 0.5
        cv2.imwrite(os.path.join(save_dir, f"CAM_{os.path.basename(image_path)}"),
                    np.uint8(overlay[:, :, ::-1] * 255))

    # 4) Insert/delete step by step
    sorted_idx = np.argsort(-cam.flatten())
    original_tensor = input_tensor[0]
    base_img = torch.zeros_like(original_tensor)
    _, H, W = original_tensor.shape

    logits_list = []
    total_pixels = H * W

    for step in range(Config.num_steps + 1):
        k = int(step / Config.num_steps * total_pixels)
        coords = np.unravel_index(sorted_idx[:k], (H, W))
        mask = torch.zeros((H, W), device=Config.device)
        mask[coords] = 1
        mask3 = mask.unsqueeze(0).repeat(3, 1, 1)

        if mode == 'insertion':
            perturbed = base_img * (1 - mask3) + original_tensor * mask3
        else:                                   # deletion
            perturbed = original_tensor * (1 - mask3) + base_img * mask3

        with torch.no_grad():
            out = model(perturbed.unsqueeze(0))
            logit = out[0, pred_class].item()   # ← raw logit
        logits_list.append(logit)

        # (Optional)Save perturbation
        if save_dir:
            img_np = perturbed.cpu().numpy()
            for c in range(3):
                img_np[c] = img_np[c] * Config.std[c] + Config.mean[c]
            img_np = np.clip(img_np.transpose(1, 2, 0), 0, 1)
            cv2.imwrite(os.path.join(
                save_dir, f'{mode}_{os.path.basename(image_path)}_step{step:02d}.png'),
                np.uint8(img_np[:, :, ::-1] * 255))

    # 5) Perform Min-Max normalization on the logits of a single graph
    logits_arr = np.array(logits_list, dtype=np.float32)
    denom = logits_arr.max() - logits_arr.min()
    if denom < 1e-6:
        if verbose:
            print(f"{os.path.basename(image_path)} - 无 logit 变化，跳过 AUC")
        return None
    scores_norm = (logits_arr - logits_arr.min()) / denom

    # 6) Calculate the AUC and plot it
    x = np.linspace(0, 1, Config.num_steps + 1)
    auc_score = np.trapz(scores_norm, x)

    if save_dir:
        plt.figure(figsize=(6, 4))
        plt.plot(x, scores_norm, label=f'{mode} curve')
        plt.xlabel('Fraction of Pixels')
        plt.ylabel('Normalized Logit')
        plt.title(f'{mode.upper()} - AUC: {auc_score:.4f} | Class: {pred_class}')
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(save_dir,
                                 f'{mode}_curve_{os.path.basename(image_path)}.png'))
        plt.close()

    if verbose:
        print(f"{os.path.basename(image_path)} - [{mode.upper()}] AUC: {auc_score:.4f}")
    return auc_score

# Evaluation the results of auc
def evaluate_dataset(dataset_dir, mode='insertion', save_output=False, verbose=False):
    image_paths = [os.path.join(dataset_dir, f)
                   for f in os.listdir(dataset_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_paths)} images in dataset.")

    model = models.resnet50(pretrained=True).to(Config.device).eval()

    cam_extractor = GradCAMElementWise(model=model, target_layers=[model.layer4])

    auc_scores = []
    for image_path in tqdm(image_paths, desc=f"Evaluating ({mode})"):
        save_dir = (os.path.join("debug_outputs", os.path.basename(image_path))
                    if save_output else None)
        auc = evaluate_single_image(model, cam_extractor, image_path,
                                    mode=mode, save_dir=save_dir, verbose=verbose)
        if auc is not None:
            auc_scores.append(auc)

    if auc_scores:
        mean_auc = np.mean(auc_scores)
        print(f"\n[FINAL] Mean AUC ({mode}): {mean_auc:.4f}")
    else:
        mean_auc = None
        print(f"\n[WARNING] No valid AUC obtained for {mode}.")
    return mean_auc

# ---------- 入口 ----------
if __name__ == '__main__':
    dataset_path = r'/home/data/data/ILSVRC/Data/CLS-LOC/val_pre2000'  # 修改为你的数据集路径
    # Evaluation Insertion
    print("Evaluating Insertion AUC...")
    ins_auc = evaluate_dataset(dataset_path, mode='insertion', save_output=False, verbose=True)

    # Evaluation Deletion
    print("\nEvaluating Deletion AUC...")
    del_auc = evaluate_dataset(dataset_path, mode='deletion', save_output=False, verbose=True)

    # Outputs
    print("\n========== FINAL RESULTS ==========")
    print(f"Insertion AUC: {ins_auc:.4f}" if ins_auc is not None else "Insertion AUC: N/A")
    print(f"Deletion AUC : {del_auc:.4f}" if del_auc is not None else "Deletion AUC : N/A")
    print("===================================")
