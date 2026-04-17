import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from preprocess_data import get_dataloaders
from draw_loss import draw_loss

PATH = "./models/beed_mlp_2.pth"

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
        
# class Net(nn.Module):
#     def __init__(self, input_size=16, num_classes=4):
#         super().__init__()
#         self.fc1 = nn.Linear(input_size, 128)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.fc2 = nn.Linear(128, 64)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.fc3 = nn.Linear(64, 32)
#         self.bn3 = nn.BatchNorm1d(32)
#         self.fc4 = nn.Linear(32, num_classes)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         x = F.gelu(self.bn1(self.fc1(x)))
#         x = self.dropout(x)
#         x = F.gelu(self.bn2(self.fc2(x)))
#         x = F.gelu(self.bn3(self.fc3(x)))
#         x = self.fc4(x)
#         return x

# class Net(nn.Module):
#     def __init__(self, input_size=16, num_classes=4):
#         super().__init__()
#         self.fc1 = nn.Linear(input_size, 128)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.fc2 = nn.Linear(128, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.fc3 = nn.Linear(256, 128)
#         self.bn3 = nn.BatchNorm1d(128)
#         self.fc4 = nn.Linear(128, 64)
#         self.bn4 = nn.BatchNorm1d(64)
#         self.fc5 = nn.Linear(64, num_classes)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         x = F.gelu(self.bn1(self.fc1(x)))
#         x = self.dropout(x)
#         x = F.gelu(self.bn2(self.fc2(x)))
#         x = F.gelu(self.bn3(self.fc3(x)))
#         x = F.gelu(self.bn4(self.fc4(x)))
#         x = self.fc5(x)
#         return x

class Net(nn.Module):
    def __init__(self, input_size=16, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc6 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.fc7 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.gelu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.gelu(self.bn2(self.fc2(x)))
        x = F.gelu(self.bn3(self.fc3(x)))
        x = F.gelu(self.bn4(self.fc4(x)))
        x = F.gelu(self.bn5(self.fc5(x)))
        x = F.gelu(self.bn6(self.fc6(x)))
        x = self.fc7(x)
        return x

def train(epochs, train_dataloader, val_dataloader, device, net=None):  
    if net is None:
        net = Net()
    
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
    average_train_losses = []
    val_losses = []
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        epoch_sum_loss = 0.0
        amount = 0
        for i, data in enumerate(train_dataloader):
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
        average_loss = epoch_sum_loss / amount
        print(f"Average loss for epoch {epoch+1}: {average_loss}")
        average_train_losses.append(average_loss)

        net.eval()
        epoch_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = epoch_val_loss / val_batches
        val_losses.append(avg_val_loss)
        
    draw_loss(average_train_losses, val_losses)
    torch.save(net.state_dict(), PATH)

def test(dataloader, classes, device, net=None):

    net = Net()
    net.load_state_dict(torch.load(PATH, weights_only=True))
    net.to(device)

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    
    net.eval()
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
    batch_size = 256
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(batch_size)

    # epochs = 1000
    # train(epochs, train_dataloader, val_dataloader, device)

    classes = (0, 1, 2, 3)
    print(f"MLP on validation dataset")
    test(val_dataloader, classes, device)
    print(f"MLP on test dataset")
    test(test_dataloader, classes, device)
    
    
