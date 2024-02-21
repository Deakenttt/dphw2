from typing import Any
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import os
from torchvision.io import read_image
from torchvision import transforms
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import torch
from torch import nn
from torch import Tensor
from tqdm import tqdm
from matplotlib import pyplot as plt


class ImageDataset( Dataset ):

    def __init__(self, is_val= False, transform = None) -> None:

        if is_val:
            self.df = pd.read_csv('validation.csv', index_col=0)
        else:
            self.df = pd.read_csv('train.csv', index_col= 0)

        self.cls_names = self.df['cls_name'].unique().tolist()
        self.df['label'] = self.df['cls_name'].apply( self.cls_names.index )
        self.transform = transform

        # Initialize OneHotEncoder
        self.onehot_encoder = OneHotEncoder(categories='auto', sparse_output=False)
        # Fit the encoder with all labels in the dataset
        labels = self.df['label'].values.reshape(-1, 1)
        self.onehot_encoder.fit(labels)

    def get_num_classes(self):
        return len( self.cls_names )

    def __len__(self):
        return len( self.df )

    def __getitem__(self, index):
        path = self.df.iloc[index]['path']
        img = read_image( path ).type( torch.float32 )
        target = self.df.iloc[index]['label']
        if self.transform is not None:
            img = self.transform( img )
        target = torch.tensor(target)
        # Convert integer label to one-hot encoding
        target_onehot = self.onehot_encoder.transform([[target]])[0]
        return img/255 , torch.tensor(target_onehot, dtype=torch.float32)
        # return img / 255, target

def collate_fn( batch ):
    imgs, targets = [], []

    for img, target in batch:
        imgs.append( img )
        targets.append( target )

    imgs = torch.stack( imgs, dim= 0 )
    targets = torch.stack( targets, dim= 0 )
    return imgs, targets


def init_weights( m ):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)



# Define the ResNet18 model
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.downsample(identity)
        out = self.relu(out)

        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=101):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2, stride=2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_epochs = 50
    batch_size = 64
    learning_rate = 0.0001
    weight_decay = 0.001

    transform = transforms.Compose([
        #transforms.Normalize( (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) ),
        transforms.RandomVerticalFlip(.5)
    ])

    train_dataset = ImageDataset(is_val=False, transform=transform)
    val_dataset = ImageDataset(is_val=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = ResNet18().to(device)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("======== start modeling ===========")
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        model.train()
        train_loss = 0.0
        avg_train_loss = 0.0
        correct = 0
        total = 0
        # Iterate through training data loader
        for i, (inputs, labels) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # one-hot to class indices
            # labels = torch.argmax(labels.squeeze(), dim=1)
            # labels = labels.squeeze().long()

            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            #predicted = outputs

            # encoder = OneHotEncoder(categories='auto', sparse_output=False)
            # predicted = encoder.fit_transform(predicted.detach().cpu().numpy().reshape(-1, 1))
            # predicted = torch.tensor(predicted, dtype=torch.float32, device=device)
            #predicted = predicted.to(device).to_dense()

            # print("predict:", predicted, " the size of predictoion is: ", predicted.shape, "data type: ", predicted.dtype, "\n",)
            # print("label:", labels, " the size of label is: ", labels.shape, "data type: ", labels.dtype)
            # os.abort()

            labels = torch.argmax(labels.squeeze(), dim=1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += (predicted == labels).sum().item()
            avg_train_loss = train_loss / len(train_dataloader)
            train_accuracy = correct / total

            # Validation
            model.eval()
            val_loss = 0.0
            avg_val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    # labels = torch.argmax(labels.squeeze(), dim=1)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    # predicted = outputs
                    loss = criterion(outputs, labels)

                    labels = torch.argmax(labels.squeeze(), dim=1)
                    val_loss += loss.item()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_val_loss = val_loss / len(val_dataloader)
            val_accuracy = correct / total
            #print(f'Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
                f'Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2%}, '
                f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2%}')

    torch.save(model.state_dict(), 'ResNet18_model.pth')