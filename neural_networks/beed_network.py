import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from preprocess_data import get_dataloaders

PATH = "./models/beed_mlp.pth"

# class Net(nn.Module):

#     def __init__(self,):
#         super().__init__()
#         self.conv1 = nn.Conv1d(16, 32, 3)
#         self.bn1 = nn.BatchNorm1d(32)
#         self.pool1 = nn.MaxPool1d(2)
#         self.conv2 = nn.Conv1d(32, 64, 3)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.pool2 = nn.MaxPool1d(2)
#         self.fc1 = nn.Linear(64 * 4, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.dropout = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(64, 4)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool1(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool2(x)
#         x = torch.flatten(x, 1)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x
        
class Net(nn.Module):
    def __init__(self, input_size=16, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def train(epochs, dataloader, device):
    net = Net()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0
        epoch_sum_loss = 0.0
        amount = 0
        for i, data in enumerate(dataloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_sum_loss += loss.item()
            running_loss += loss.item()
            amount += 1
            if i % 10 == 9:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}")
                running_loss = 0.0
        print(f"Average loss for epoch {epoch+1}: {epoch_sum_loss / amount}")
    torch.save(net.state_dict(), PATH)
    print("Finish Training")

def test(dataloader, classes, device):
    net = Net()
    net.load_state_dict(torch.load(PATH, weights_only=True))
    net.to(device)

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    sum_acc = 0.0
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname} is {accuracy:.1f} %')
        sum_acc += accuracy
    print(f"Average accuracy: {sum_acc / 4}")

if __name__ == "__main__":
    batch_size = 128
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(batch_size)

    epochs = 1000
    train(epochs, train_dataloader, device)

    classes = (0, 1, 2, 3)
    test(test_dataloader, classes, device)
    
    
