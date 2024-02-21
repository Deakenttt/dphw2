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
from tqdm import tqdm
from matplotlib import pyplot as plt


class ImageDataset( Dataset ):

    def __init__(self, is_val= False, transform = None) -> None:

        if is_val:
            self.df = pd.read_csv( 'validation.csv', index_col=0 )
        else:
            self.df = pd.read_csv( 'train.csv', index_col= 0 )

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

        # Apply softmax to get class probabilities
        # probabilities = F.softmax(x, dim=1)

        # Convert to one-hot encoding
        # _, predicted = torch.max(x,1)
        # one_hot_output = F.one_hot(predicted, num_classes=101)
        # return one_hot_output
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

    model = LeNet5().to(device)
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

    torch.save(model.state_dict(), 'lenet5_model.pth')