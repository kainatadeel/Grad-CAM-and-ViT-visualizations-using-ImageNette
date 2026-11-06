
import torch
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights

from src.dataset import load_dataset
from src.models import load_models


def evaluate_model(model, dataloader, device, class_names, imagenet_labels):
    """Calculate overall and per-category classification accuracy."""

    model.eval()

    total_correct = 0
    total_images = 0

    # Track correct predictions for each ImageNette category.
    category_correct = {name: 0 for name in class_names}
    category_total = {name: 0 for name in class_names}

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)

            # Get the model's predicted ImageNet class.
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            # Convert ImageNette labels into their corresponding
            # ImageNet class indices for a fair comparison.
            expected_indices = torch.tensor(
                [imagenet_labels[class_names[label]]
                 for label in labels.tolist()],
                device=device
            )

            correct = predictions == expected_indices

            total_correct += correct.sum().item()
            total_images += labels.size(0)

            # Record results for each category.
            for i, label in enumerate(labels.tolist()):
                category = class_names[label]

                category_total[category] += 1
                category_correct[category] += correct[i].item()

    accuracy = (total_correct / total_images) * 100

    return (
        total_correct,
        total_images,
        accuracy,
        category_correct,
        category_total
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load the complete ImageNette validation dataset.
    _, val_loader = load_dataset(batch_size=32)

    class_names = val_loader.dataset.classes

    # Map ImageNette folder names to their ImageNet category names.
    imagenette_to_imagenet = {
        "n01440764": "tench",
        "n02102040": "English springer",
        "n02979186": "cassette player",
        "n03000684": "chain saw",
        "n03028079": "church",
        "n03394916": "French horn",
        "n03417042": "garbage truck",
        "n03425413": "gas pump",
        "n03445777": "golf ball",
        "n03888257": "parachute"
    }

    # Load the ImageNet category lists.
    resnet_labels = ResNet18_Weights.DEFAULT.meta["categories"]
    vit_labels = ViT_B_16_Weights.DEFAULT.meta["categories"]

    # Find the ImageNet output index for each ImageNette category.
    resnet_indices = {
        folder: resnet_labels.index(imagenette_to_imagenet[folder])
        for folder in class_names
    }

    vit_indices = {
        folder: vit_labels.index(imagenette_to_imagenet[folder])
        for folder in class_names
    }

    # Load pretrained models.
    resnet, vit = load_models()

    resnet = resnet.to(device)
    vit = vit.to(device)

    print("\nEvaluating ResNet18...")
    resnet_results = evaluate_model(
        resnet, val_loader, device, class_names, resnet_indices
    )

    print("\nEvaluating ViT...")
    vit_results = evaluate_model(
        vit, val_loader, device, class_names, vit_indices
    )

    # Display overall results.
    print("\n========== OVERALL RESULTS ==========")

    print(
        f"ResNet18: {resnet_results[0]}/{resnet_results[1]} "
        f"correct ({resnet_results[2]:.2f}%)"
    )

    print(
        f"ViT: {vit_results[0]}/{vit_results[1]} "
        f"correct ({vit_results[2]:.2f}%)"
    )

    # Display accuracy for each category.
    print("\n========== CATEGORY RESULTS ==========")

    print(
        f"{'Category':<20} {'ResNet18':<15} {'ViT':<15}"
    )

    for category in class_names:
        resnet_accuracy = (
            resnet_results[3][category]
            / resnet_results[4][category]
        ) * 100

        vit_accuracy = (
            vit_results[3][category]
            / vit_results[4][category]
        ) * 100

        readable_name = imagenette_to_imagenet[category]

        print(
            f"{readable_name:<20} "
            f"{resnet_accuracy:<15.2f} "
            f"{vit_accuracy:<15.2f}"
        )

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
