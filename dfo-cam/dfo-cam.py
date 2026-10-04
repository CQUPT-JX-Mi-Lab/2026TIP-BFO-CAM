import torch
import torch.nn.functional as F
import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM

# Using the initial weights by Grad-CAM
def init_weights_by_gradcam(activations: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
    """
    Apply GAP, normalize and apply ReLU activation to the gradient
    """
    with torch.no_grad():
        weights = torch.mean(grads, dim=(2, 3))  # GAP: (B, C)
        max_vals = torch.max(torch.abs(weights), dim=1, keepdim=True)[0]  # 防止除以 0
        max_vals = torch.where(max_vals < 1e-7, torch.ones_like(max_vals), max_vals)
        weights = weights / max_vals  # Weight normalization
        weights = F.relu(weights)
    return weights  # shape: (B, C)

# Using the initial weights by Weight of 0.5
def init_weights_all_zerofive(activations: torch.Tensor) -> torch.Tensor:
    B, C, _, _ = activations.shape
    return torch.full((B, C), 0.5, device=activations.device)

# Using the initial weights by Weight of random
def init_weights_random(activations: torch.Tensor) -> torch.Tensor:
    B, C, _, _ = activations.shape
    return torch.randn(B, C, device=activations.device) * 0.1

# Using the initial weights by Weight of ShapleyCAM
def init_weights_shapleycam(activations: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
    activations = activations.detach().requires_grad_(True)
    grads = grads.detach()
    scalar_grads = torch.sum(grads * activations)
    try:
        hvp = torch.autograd.grad(
            outputs=scalar_grads,
            inputs=activations,
            retain_graph=False,
            create_graph=False
        )[0]
    except:
        hvp = torch.zeros_like(activations)
    grads_std = (grads - grads.mean()) / (grads.std() + 1e-6)
    hvp_std = (hvp - hvp.mean()) / (hvp.std() + 1e-6)
    shapley_weights = grads_std - 0.5 * hvp_std
    return shapley_weights.mean(dim=(2, 3)).detach()  # (B, C)

# Using other original weights as you want >>>
# ...


class DFOCAM(BaseCAM):
    def __init__(self, model, target_layers, reshape_transform=None, init_method="shapleycam"):
        super(DFOCAM, self).__init__(
            model,
            target_layers,
            reshape_transform=reshape_transform,
            uses_gradients=True
        )
        self.init_method = init_method  # Optional: gradcam / zerofive / random / shapleycam or others...

    def get_cam_weights(
        self,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        targets: list,
        activations: torch.Tensor,
        grads: torch.Tensor,
    ) -> np.ndarray:
        """
        activations: (B, C, H, W)
        grads: (B, C, H, W)
        """
        device = input_tensor.device
        # 🔧 Convert numpy arrays to torch tensors
        grads = torch.from_numpy(grads).to(device)
        activations = torch.from_numpy(activations).to(device)
        B, C, H, W = activations.shape

        # Select the strategy based on the initialization weight
        if self.init_method == "gradcam":
            init_weights = init_weights_by_gradcam(activations, grads)
        elif self.init_method == "zerofive":
            init_weights = init_weights_all_zerofive(activations)
        elif self.init_method == "random":
            init_weights = init_weights_random(activations)
        elif self.init_method == "shapleycam":
            init_weights = init_weights_shapleycam(activations, grads)
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")

        weights = init_weights.clone().detach().requires_grad_(True)

        # Set the optimized hyperparameters as Epochs or Learning-Rate
        epochs = 50
        lr = 0.01

        # Set up an optimizer to optimize the weights
        optimizer = torch.optim.AdamW([weights], lr=lr)

        # Save the optimal result obtained during the optimization process
        best_loss = float("inf")
        best_weights = weights.detach().clone()

        # Weight Optimization Process
        for epoch in range(epochs):  # epoch = 50

            optimizer.zero_grad()

            # The weighted sum is used to obtain the saliency map (B, H, W)
            saliency_map = torch.einsum('bc,bchw->bhw', weights, activations)

            saliency_map = F.relu(saliency_map)

            eps = 1e-10  # Prevent Division by Zero
            saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + eps)

            # Adjust the size of the saliency_map
            saliency_map = saliency_map.unsqueeze(1)  # (B, 1, H, W)

            # Upsample to image size
            saliency_map_upsampled = torch.nn.functional.interpolate(
                saliency_map,
                size=input_tensor.shape[2:],
                mode='bilinear',
                align_corners=False
            )

            # Broadcast significant map to 3 channels
            saliency_map_upsampled = saliency_map_upsampled.expand(-1, 3, -1, -1)

            # Generate forward and backward input images
            pos_input = input_tensor * saliency_map_upsampled
            neg_input = input_tensor * (1 - saliency_map_upsampled)

            # Obtain the logit scores for the mask and anti-mask inputs
            pos_logits = self.model(pos_input)
            neg_logits = self.model(neg_input)

            # Extract the index of the target category
            target_idx = torch.tensor([t.category for t in targets], device=pos_logits.device)  # (B,)

            # Obtain softmax probabilities
            pos_probs = F.softmax(pos_logits, dim=1)  # (B, num_classes)
            neg_probs = F.softmax(neg_logits, dim=1)

            # Obtain the softmax scores of the original image
            with torch.no_grad():
                original_logits = self.model(input_tensor)
                original_probs = F.softmax(original_logits, dim=1)
                original_score = original_probs[range(len(target_idx)), target_idx]

            # Obtain the probability of the target category
            pos_score = pos_probs[range(len(target_idx)), target_idx]  # (B,)
            neg_score = neg_probs[range(len(target_idx)), target_idx]

            # Optimize using smooth_l1_loss on dual-flow
            loss_forward = F.smooth_l1_loss(pos_score, torch.ones_like(original_score))
            loss_backward = F.smooth_l1_loss(neg_score, torch.zeros_like(neg_score))

            # Combine the two losses as the objective function
            loss = loss_forward + loss_backward

            # If you want to view the changes in loss during the optimization process, please un-comment
            # print(f"[Epoch {epoch + 1}/{epochs}] Loss_forward:{loss_forward.item():.4f} Loss_backward:{loss_backward.item():.4f} Loss:{loss.item():.4f}")
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_weights = weights.detach().clone()

        final_weights = best_weights.detach().cpu().numpy()

        # Interpretation Process is carried out by the BaseCAM library.
        return final_weights