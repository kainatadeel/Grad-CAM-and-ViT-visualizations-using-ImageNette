
import os
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights

from src.dataset import load_dataset
from src.models import load_models


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def generate_predictions(model, dataloader, device, imagenet_labels, class_names):
    """
    Collect actual ImageNette labels and model predictions.

    Predictions outside the ten ImageNette categories are
    grouped into an additional 'Other ImageNet class' column.
    """

    # Map ImageNette categories to their ImageNet class indices.
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

    # Convert ImageNet category names into output indices.
    imagenet_indices = {
        folder: imagenet_labels.index(imagenette_to_imagenet[folder])
        for folder in class_names
    }

    # Reverse the mapping to identify ImageNette predictions.
    imagenet_to_imagenette = {
        imagenet_indices[folder]: index
        for index, folder in enumerate(class_names)
    }

    actual_labels = []
    predicted_labels = []

    model.eval()

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().tolist()

            # Actual labels use ImageNette's category indices.
            actual_labels.extend(labels.tolist())

            # Convert model predictions to ImageNette categories.
            # Predictions outside the ten categories become class 10.
            for prediction in predictions:
                mapped_prediction = imagenet_to_imagenette.get(
                    prediction, 10
                )
                predicted_labels.append(mapped_prediction)

    return actual_labels, predicted_labels


def save_confusion_matrix(actual, predicted, display_names, filename, title):
    """Generate and save a labeled confusion matrix."""

    labels = list(range(len(display_names)))

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=labels
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=display_names
    )

    fig, ax = plt.subplots(figsize=(12, 10))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        xticks_rotation=45,
        colorbar=True
    )

    ax.set_title(title)
    ax.set_xlabel("Predicted Category")
    ax.set_ylabel("Actual Category")

    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, filename)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load the complete validation dataset.
    _, val_loader = load_dataset(batch_size=32)
    class_names = val_loader.dataset.classes

    # Convert folder names into readable category names.
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

    display_names = [
        imagenette_names[folder] for folder in class_names
    ]

    # Add a category for predictions outside ImageNette's ten classes.
    display_names.append("Other ImageNet class")

    # Load both pretrained models.
    resnet, vit = load_models()

    resnet = resnet.to(device)
    vit = vit.to(device)

    # Obtain the ImageNet class labels for each model.
    resnet_labels = ResNet18_Weights.DEFAULT.meta["categories"]
    vit_labels = ViT_B_16_Weights.DEFAULT.meta["categories"]

    # Generate predictions for ResNet18.
    print("\nGenerating ResNet18 predictions...")

    resnet_actual, resnet_predicted = generate_predictions(
        resnet,
        val_loader,
        device,
        resnet_labels,
        class_names
    )

    # Generate predictions for ViT.
    print("\nGenerating ViT predictions...")

    vit_actual, vit_predicted = generate_predictions(
        vit,
        val_loader,
        device,
        vit_labels,
        class_names
    )

    # Generate and save both confusion matrices.
    print("\nCreating confusion matrices...")

    save_confusion_matrix(
        resnet_actual,
        resnet_predicted,
        display_names,
        "resnet18_confusion_matrix.png",
        "ResNet18 Confusion Matrix"
    )

    save_confusion_matrix(
        vit_actual,
        vit_predicted,
        display_names,
        "vit_confusion_matrix.png",
        "ViT Confusion Matrix"
    )

    print("\nBoth confusion matrices completed!")


if __name__ == "__main__":
    main()
