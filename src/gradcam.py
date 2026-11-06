
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class GradCAM:
    """
    Generate Grad-CAM heatmaps to visualize which image regions
    contribute to a CNN's prediction.
    """

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # Register hooks to capture feature maps and gradients.
        self.forward_hook = target_layer.register_forward_hook(
            self.save_activations
        )

        self.backward_hook = target_layer.register_full_backward_hook(
            self.save_gradients
        )

    def save_activations(self, module, input, output):
        # Capture the feature maps produced by the target layer.
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        # Capture gradients flowing backward through the target layer.
        self.gradients = grad_output[0].detach()

    def generate(self, image, class_idx=None):

        # Clear previous gradients before calculating a new explanation.
        self.model.zero_grad()

        # Get the model's class predictions.
        output = self.model(image)

        # Explain the predicted class unless another class is specified.
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Calculate gradients for the selected class score.
        score = output[:, class_idx]
        score.backward()

        # Average gradients across spatial dimensions to get channel weights.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Combine feature maps using their gradient-based importance weights.
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # Keep positive contributions and resize the heatmap to image size.
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=image.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        # Normalize heatmap values between 0 and 1.
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, class_idx

    def remove_hooks(self):
        # Remove hooks when Grad-CAM is no longer needed.
        self.forward_hook.remove()
        self.backward_hook.remove()
