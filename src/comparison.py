import os
import torch
import matplotlib.pyplot as plt

from torchvision.models import ResNet18_Weights, ViT_B_16_Weights

from src.dataset import load_dataset
from src.models import load_models
from src.gradcam import GradCAM
from src.vit_visualization import generate_attention_map


# Folder for saving generated comparison images.
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# Convert ImageNet-normalized tensors back into displayable images.
def denormalize(image):

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = image.detach().cpu() * std + mean

    return image.permute(1, 2, 0).clamp(0, 1).numpy()


def main():

    # Use GPU if available; otherwise, run on CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load one validation image for both models.
    _, val_loader = load_dataset(batch_size=1)

    images, labels = next(iter(val_loader))
    images = images.to(device)

    # Get the actual ImageNette category.
    class_names = val_loader.dataset.classes
    true_label = class_names[labels[0].item()]

    # Convert ImageNette folder IDs into readable category names.
    imagenette_names = {
        "n01440764": "Tench",
        "n02102040": "English Springer",
        "n02979186": "Cassette Player",
        "n03000684": "Chain Saw",
        "n03028079": "Church",
        "n03394916": "French Horn",
        "n03417042": "Garbage Truck",
        "n03425413": "Gas Pump",
        "n03445777": "Golf Ball",
        "n03888257": "Parachute"
    }

    true_label_name = imagenette_names.get(true_label, true_label)

    # Load both pretrained models.
    resnet, vit = load_models()

    resnet = resnet.to(device)
    vit = vit.to(device)

    # Get the human-readable ImageNet class labels.
    resnet_labels = ResNet18_Weights.DEFAULT.meta["categories"]
    vit_labels = ViT_B_16_Weights.DEFAULT.meta["categories"]

    # Get predictions from both models.
    with torch.no_grad():

        resnet_output = resnet(images)
        vit_output = vit(images)

        resnet_prediction = resnet_output.argmax(dim=1).item()
        vit_prediction = vit_output.argmax(dim=1).item()

        resnet_confidence = torch.softmax(
            resnet_output, dim=1
        )[0, resnet_prediction].item()

        vit_confidence = torch.softmax(
            vit_output, dim=1
        )[0, vit_prediction].item()

    # Generate Grad-CAM using ResNet18's final convolutional layer.
    gradcam = GradCAM(resnet, resnet.layer4[-1])

    try:
        gradcam_map, _ = gradcam.generate(
            images,
            class_idx=resnet_prediction
        )
    finally:
        gradcam.remove_hooks()

    # Generate attention rollout visualization for ViT.
    attention_map = generate_attention_map(vit, images)

    # Prepare the original image for display.
    original_image = denormalize(images[0])

    # Print the actual label and both model predictions.
    print("\n--- Classification Results ---")
    print("Actual ImageNette category:", true_label_name)

    print(
        "ResNet18 prediction:",
        resnet_labels[resnet_prediction],
        f"({resnet_confidence:.2%} confidence)"
    )

    print(
        "ViT prediction:",
        vit_labels[vit_prediction],
        f"({vit_confidence:.2%} confidence)"
    )

    # Create a side-by-side comparison of both explanations.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_image)
    axes[0].set_title(f"Original Image\nActual: {true_label_name}")

    axes[1].imshow(original_image)
    axes[1].imshow(gradcam_map, cmap="jet", alpha=0.5)
    axes[1].set_title(
        f"ResNet18 - Grad-CAM\n{resnet_labels[resnet_prediction]}"
    )

    axes[2].imshow(original_image)
    axes[2].imshow(attention_map, cmap="jet", alpha=0.5)
    axes[2].set_title(
        f"ViT - Attention Rollout\n{vit_labels[vit_prediction]}"
    )

    # Remove axis markings for cleaner visualizations.
    for axis in axes:
        axis.axis("off")

    plt.tight_layout()

    # Save the comparison image for the project report.
    output_path = os.path.join(RESULTS_DIR, "comparison.png")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("\nComparison saved to:", output_path)


if __name__ == "__main__":
    main()
