import torch
import torch.nn as nn
from models.modelv1 import ModelV1
from torchsummary import summary
from dataset import get_train, get_test
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import random


# get random image from test
def get_random_image_from_test(test_data, n=1):
    random_indices = random.sample(range(len(test_data)), n)
    images = [test_data[i][0] for i in random_indices]
    return images

def plot_images(images, n=1):
    fig, axes = plt.subplots(1, n, figsize=(n * 3, 3))
    for i, img in enumerate(images):
        if n == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.imshow(img.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        ax.axis('off')
    plt.show()

# load model
def load_model(model_path, device):
    model = ModelV1(in_shape=1, hidden=128, n_classes=7).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

# predict and plot with true vs predicted labels
def predict_and_plot(model, test_data, n=3, device='gpu'):
    model.eval()
    images = get_random_image_from_test(test_data, n)
    images_tensor = torch.stack(images).to(device)

    with torch.no_grad():
        outputs = model(images_tensor)
        _, predicted = torch.max(outputs, 1)

    # Plot images with predicted labels
    plot_images(images_tensor.cpu(), n)
    print("Predicted labels:", predicted.cpu().numpy())
if __name__ == "__main__":
    print(f'Running on torch version: {torch.__version__}')

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Load datasets
    train_data = get_train()
    test_data = get_test()

    # Initialize model
    model = ModelV1(in_shape=train_data[0][0].shape[0],
                    hidden=128,
                    n_classes=len(train_data.classes)).to(device)

    # Load model weights
    model_path = 'modelv1.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Predict and plot images
    predict_and_plot(model, test_data, n=3, device=device)