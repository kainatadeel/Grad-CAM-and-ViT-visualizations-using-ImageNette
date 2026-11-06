
import os
import torch
import matplotlib.pyplot as plt

from torchvision.models import ResNet18_Weights, ViT_B_16_Weights
from src.dataset import load_dataset
from src.models import load_models
from src.gradcam import GradCAM
from src.vit_visualization import generate_attention_map

# Create the results folder if it does not already exist.
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def denormalize(image):
    """Convert a normalized image tensor back into a displayable image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = image.detach().cpu() * std + mean
    return image.permute(1, 2, 0).clamp(0, 1).numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load the ImageNette validation dataset.
    _, val_loader = load_dataset(batch_size=1)
    class_names = val_loader.dataset.classes

    # Convert ImageNette folder names into readable category names.
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

    # Load both pretrained classification models.
    resnet, vit = load_models()
    resnet = resnet.to(device)
    vit = vit.to(device)

    resnet_labels = ResNet18_Weights.DEFAULT.meta["categories"]
    vit_labels = ViT_B_16_Weights.DEFAULT.meta["categories"]

    # Select one image from each of five different categories.
    selected_images = {}

    for images, labels in val_loader:
        category = class_names[labels[0].item()]

        if category not in selected_images:
            selected_images[category] = images

        if len(selected_images) == 5:
            break

    # Process each selected image.
    for index, (category, images) in enumerate(
        selected_images.items(), start=1
    ):
        images = images.to(device)

        true_label = imagenette_names.get(category, category)

        # Generate predictions using both models.
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

        # Generate Grad-CAM visualization for ResNet18.
        gradcam = GradCAM(resnet, resnet.layer4[-1])

        try:
            gradcam_map, _ = gradcam.generate(
                images,
                class_idx=resnet_prediction
            )
        finally:
            gradcam.remove_hooks()

        # Generate Attention Rollout visualization for ViT.
        attention_map = generate_attention_map(vit, images)

        original_image = denormalize(images[0])

        # Print classification results.
        print(f"\n--- Image {index} ---")
        print("Actual category:", true_label)

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

        # Display the original image and both explanations.
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original_image)
        axes[0].set_title(f"Original Image\nActual: {true_label}")

        axes[1].imshow(original_image)
        axes[1].imshow(gradcam_map, cmap="jet", alpha=0.5)
        axes[1].set_title(
            f"ResNet18 - Grad-CAM\n"
            f"{resnet_labels[resnet_prediction]}"
        )

        axes[2].imshow(original_image)
        axes[2].imshow(attention_map, cmap="jet", alpha=0.5)
        axes[2].set_title(
            f"ViT - Attention Rollout\n"
            f"{vit_labels[vit_prediction]}"
        )

        for axis in axes:
            axis.axis("off")

        plt.tight_layout()

        # Save each comparison separately.
        output_path = os.path.join(
            RESULTS_DIR,
            f"comparison_{index}.png"
        )

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print("Saved:", output_path)

    print("\nAll five comparisons completed!")


if __name__ == "__main__":
    main()
