import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms

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

# Load the pre-trained LeNet-5 model
model = LeNet5()

# Load the weights (adjust the path based on where your model is saved)
model.load_state_dict(torch.load('lenet5_model.pth'))

# Set the model to evaluation mode
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Load the images
accordion_image = transform(Image.open('Dataset/train/accordion/image_0001.jpg')).unsqueeze(0)
camera_image = transform(Image.open('Dataset/train/camera/image_0001.jpg')).unsqueeze(0)


# Function to visualize layer activations using torchvision
def visualize_activations_torchvision(model, image, layer_num):
    activation = None
    hooks = []

    def hook_fn(module, input, output):
        nonlocal activation
        activation = output.detach()

    # Find the layer by name and register the hook
    target_layer = None
    for name, layer in model.named_children():
        if name == layer_num:
            target_layer = layer
            break

    if target_layer is not None:
        hook = target_layer.register_forward_hook(hook_fn)
        hooks.append(hook)

        with torch.no_grad():
            model(image)

        # Remove the hook
        hook.remove()

    # Plot the activations for each filter in a row
    if activation is not None:
        activation = activation.squeeze(0)
        num_filters = activation.size(0)

        grid_image = torchvision.utils.make_grid(activation.unsqueeze(1), nrow=num_filters)

        plt.imshow(grid_image.permute(1, 2, 0), cmap='viridis')
        plt.title(f'{layer_num} Activations')
        plt.axis('off')
        plt.show()




# Visualize activations for the first convolutional layer
visualize_activations_torchvision(model, accordion_image, layer_num=0)
visualize_activations_torchvision(model, camera_image, layer_num=0)

# Visualize activations for the second convolutional layer
visualize_activations_torchvision(model, accordion_image, layer_num=3)
visualize_activations_torchvision(model, camera_image, layer_num=3)