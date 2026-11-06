
import torch
import torchvision.models as models


def load_models():
    """
    Load pretrained ResNet18 and Vision Transformer models.

    Both models use ImageNet pretrained weights, allowing us
    to study their predictions and visual explanations without
    training the architectures from scratch.
    """

    # Load ResNet18, a CNN that uses residual connections
    # to learn visual features through convolutional layers.
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Load ViT-B/16, a Vision Transformer that divides images
    # into 16x16 patches and processes them using self-attention.
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

    # Set both models to evaluation mode.
    # This disables training-specific behavior during inference.
    resnet.eval()
    vit.eval()

    return resnet, vit


if __name__ == "__main__":

    resnet, vit = load_models()

    print("Models loaded successfully!")
    print("ResNet18 parameters:", sum(p.numel() for p in resnet.parameters()))
    print("ViT parameters:", sum(p.numel() for p in vit.parameters()))
