import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Location of the ImageNette dataset downloaded through FastAI.
# The dataset contains separate training and validation folders.
DATA_DIR = os.path.expanduser(
    "~/.fastai/data/imagenette2-160"
)

# Preprocess images so they have a consistent format and size
# before being passed into the pretrained deep learning models.
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match model input dimensions
    transforms.ToTensor(),         # Convert images into PyTorch tensors
    transforms.Normalize(           # Normalize using ImageNet channel statistics
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_dataset(batch_size=32):
    """
    Load the ImageNette training and validation datasets.

    ImageFolder automatically assigns labels based on the
    category subfolder names. DataLoader groups images into
    batches for efficient processing by the models.
    """

    # Load training images and apply the preprocessing steps.
    train_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, "train"),
        transform=transform
    )

    # Load validation images using the same preprocessing.
    # Shuffling is disabled later so evaluation is consistent.
    val_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, "val"),
        transform=transform
    )

    # Create batches of images for model processing.
    # Training images are shuffled to vary their order.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Validation images are not shuffled because their
    # order does not need to change during evaluation.
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader

# Run this section only when this file is executed directly.
# It checks that the dataset loads and displays basic information.
if __name__ == "__main__":

    train_loader, val_loader = load_dataset()

    print("Dataset loaded successfully!")
    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))
    print("Classes:", train_loader.dataset.classes)