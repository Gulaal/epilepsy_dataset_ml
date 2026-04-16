import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split

class Net(nn.Module):

    def __init__(self,):
        super().__init__()
        self.conv1 = nn.Conv1d(16, 32, 3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
        

def train(train_X, train_y, device):
    net = Net()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)



if __name__ == "__main__":
    
    beed_file_path = "../BEED_Data.csv"
    beed_data = pd.read_csv(beed_file_path)

    X = beed_data[:, :-1]
    y = beed_data[:, -1]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 32

    train_X, train_y, test_X, test_y = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    
    train(train_X, train_y, device)
