import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' or 'Agg'
import torch
import numpy as np
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import pandas as pd
from torchvision.io import read_image
import torchvision.transforms.functional as TF

# run with this command python3 visiualization.py

class LeNet5(nn.Module):
    def __init__(self, num_classes=101):
        super(LeNet5, self).__init__()

        # Layer 1: Convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=4, stride=4)

        # Layer 2: Convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=4)

        # Layer 3: Fully Connected ReLU layer
        self.fc1 = nn.Linear(64 * 4 * 4, 1021)
        self.relu3 = nn.ReLU()

        # Layer 4: Fully Connected ReLU layer
        self.fc2 = nn.Linear(1021, 84)
        self.relu4 = nn.ReLU()

        # Layer 5: Fully Connected Softmax layer
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

class ImageDataset( Dataset ):
    def __init__(self, image_path, image_file, transform=None):
        self.image_path = image_path
        self.image_file = image_file
        self.transform = transform

    def __len__(self):
        return 1  # We have only one image

    def __getitem__(self, idx):
        image = Image.open(f"{self.image_path}/{self.image_file}")

        if self.transform:
            image = self.transform(image)
        return image


def collate_fn( batch ):
    imgs, targets = [], []
    for img, target in batch:
        imgs.append( img )
        targets.append( target )
    imgs = torch.stack( imgs, dim= 0 )
    targets = torch.stack( targets, dim= 0 )
    return imgs, targets


def conv_max_layer_plot(nrows, ncols, title, image, figsize=(16, 8), color='gray'):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 8))
    fig.suptitle(title)

    for i in range(nrows * ncols):
        image_plot = axs[i // 8, i % 8].imshow(image[0, :, :, i], cmap=color)
        axs[i // 8, i % 8].axis('off')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(image_plot, cax=cbar_ax)
    plt.show()

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_epochs = 50
    batch_size = 1
    learning_rate = 0.0001
    weight_decay = 0.001

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize( (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) ),
        transforms.RandomVerticalFlip(.5),

    ])

    # Set your image path and desired image file
    image_path = "Dataset/train/accordion"
    image_file = "image_0001.jpg"
    dataset1 = ImageDataset(image_path, image_file, transform=transform)
    image_path = "Dataset/train/camera"
    image_file = "image_0001.jpg"
    dataset2 = ImageDataset(image_path, image_file, transform=transform)

    dataset1 = DataLoader(dataset1, batch_size=batch_size, shuffle=False)
    dataset2 = DataLoader(dataset2, batch_size=batch_size, shuffle=False)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    tmp = Image.open(str("Dataset/train/camera/image_0001.jpg"))
    plt.imshow(tmp)

    # images = train_dataloader.dataset
    images = next(iter(dataset2))
    images = images.squeeze()
    print("DONE")
    #images = next(iter(train_dataloader))

    # images, labels = dataset1[1]
    # Get one batch of Data
    # images, labels = next(iter(train_dataloader))
    # Use transpose instead of reshape.
    # print(images.shape)
    images = images.numpy().transpose((1, 2, 0))
    # plt.imshow(images.squeeze(), cmap='gray')
    print(images.shape)
    # plt.show()

    net = LeNet5()
    # images = images.unsqueeze(0)
    # Get the image from the DataLoader and unsqueeze
    images = next(iter(dataset2)).squeeze()
    images =images.unsqueeze(0)
    print(images.shape)

    # Get the output of the first convolutional layer
    conv_output = net.conv1(images)
    # Rearrange dimensions and convert to numpy array
    conv_output_image = conv_output.permute(0, 2, 3, 1).detach().numpy()
    print("\n\n", conv_output.shape)
    conv_max_layer_plot(nrows=4, ncols=8, title='First Conv2D', image=conv_output_image)

    relu_1_output = net.relu1(conv_output)
    # MaxPool Layer output visualize
    max_pool_output_1 = net.maxpool1(relu_1_output)
    # Rearrange dimensions and convert to numpy array
    max_pool_output_image = max_pool_output_1.permute(0, 2, 3, 1).detach().numpy()
    print('\n\n', max_pool_output_1.shape)
    conv_max_layer_plot(nrows=4, ncols=8, title='After First ReLu', image=max_pool_output_image)

    # Get the second convulation output and visualize them
    conv_output_2 = net.conv2(max_pool_output_1)

    # Rearrange dimensions and convert to numpy array
    conv_output_2_image = conv_output_2.permute(0, 2, 3, 1).detach().numpy()
    print('\n\n', conv_output_2.shape)
    conv_max_layer_plot(nrows=8, ncols=8, title="Second Conv2D", image=conv_output_2_image)

    relu_2_output = net.relu2(conv_output_2)

    # MaxPool Layer output visualize
    max_pool_output_2 = net.maxpool2(relu_2_output)
    # Rearrange dimensions and convert to numpy array
    max_pool_output2_image = max_pool_output_2.permute(0, 2, 3, 1).detach().numpy()
    print('\n\n', max_pool_output_2.shape)
    conv_max_layer_plot(nrows=8, ncols=8, title='After Second ReLu', image=max_pool_output2_image)

