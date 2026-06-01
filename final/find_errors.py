import numpy as np
import torch
from beed_network import Net
from preprocess_data import get_dataloaders

PATH = "./models/beed_mlp.pth"
model = Net()
model.load_state_dict(torch.load(PATH, weights_only=True))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

batch_size = 64
train_loader, val_loader, test_loader = get_dataloaders(batch_size)

classes = ['0', '1', '2', '3']

model.eval()
all_preds = []
all_labels = []
all_inputs = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_inputs.append(inputs.cpu())

X_test_tensor = torch.cat(all_inputs, dim=0)

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

incorrect_idx = np.where(all_preds != all_labels)[0]
correct_idx = np.where(all_preds == all_labels)[0]

print(f"Всего примеров: {len(all_labels)}")
print(f"Правильных: {len(correct_idx)} ({100*len(correct_idx)/len(all_labels):.1f}%)")
print(f"Ошибочных: {len(incorrect_idx)}")

print("\n=== Ошибочные предсказания (первые 5) ===")
for i in incorrect_idx[:5]:
    print(f"Индекс в тестовой выборке: {i}, истинный класс: {all_labels[i]}, предсказанный: {all_preds[i]}")
    print(f"Признаки (первые 5): {X_test_tensor[i, :5].tolist()}")
    print(f"Признаки (все 16): {X_test_tensor[i].tolist()}")
    print()

print("\n=== Правильные предсказания (первые 5) ===")
for i in correct_idx[:5]:
    print(f"Индекс: {i}, истинный: {all_labels[i]}, предсказанный: {all_preds[i]}")