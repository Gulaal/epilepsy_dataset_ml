import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class BEEDDataset(Dataset):

    def __init__(self, features, labels):    
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
def read_info():
    beed_file_path = "BEED_Data.csv"
    beed_data = pd.read_csv(beed_file_path)
    X = beed_data.iloc[:, :-1]
    y = beed_data.iloc[:, -1]
    return X, y

def preprocess_info(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=3, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=3, stratify=y_temp
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.long).squeeze()

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.long).squeeze()

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.long).squeeze()

    train_dataset = BEEDDataset(X_train_t, y_train_t)
    val_dataset = BEEDDataset(X_val_t, y_val_t)
    test_dataset = BEEDDataset(X_test_t, y_test_t)

    return train_dataset, val_dataset, test_dataset

def get_dataloaders(batch_size):
    X, y = read_info()
    train_dataset, val_dataset, test_dataset = preprocess_info(X, y)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_dataloader, val_dataloader, test_dataloader
