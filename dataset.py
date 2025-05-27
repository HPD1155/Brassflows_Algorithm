import PIL.Image
import PIL.ImagePath
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import random
import PIL
from pathlib import Path



# transforms for image
data_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

def plot_transformed_image(image, transform, n=3, seed=42):
    random.seed(seed)
    random_image_paths = random.sample(image, n)
    for image_path in random_image_paths:
        with PIL.Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title('Original Image')
            ax[0].axis('off')

            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title('Transformed Image')
            ax[1].axis('off')
    plt.show()

def get_train():
    return datasets.ImageFolder(root='data/train',
                                transform=data_transforms,
                                target_transform=None)

def get_test():
    return datasets.ImageFolder(root='data/test',
                                transform=data_transforms)

if __name__ == "__main__":
    print(f'Running on torch version: {torch.__version__}')

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    image_path = Path('data/train')
    image_path_list = list(image_path.glob("*/*.jpg"))


    plot_transformed_image(
        image=image_path_list,
        transform=data_transforms,
        n=3,
        seed=42
    )
    print(f"Train data:\n{get_train()}\nTest data:\n{get_test()}")

    print(f"Classes: {get_train().class_to_idx}")

    
    img, label = get_train()[500][0], get_train()[500][1]
    print(f"Image tensor:\n{img}")
    print(f"Image shape: {img.shape}")
    print(f"Image datatype: {img.dtype}")
    print(f"Image label: {label}")
    print(f"Label datatype: {type(label)}")
    img_permute = img.permute(1, 2, 0)

    # Print out different shapes (before and after permute)
    print(f"Original shape: {img.shape} -> [color_channels, height, width]")
    print(f"Image permute shape: {img_permute.shape} -> [height, width, color_channels]")

    # Plot the image
    plt.figure(figsize=(10, 7))
    plt.imshow(img.permute(1, 2, 0))
    plt.axis("off")
    plt.show()
