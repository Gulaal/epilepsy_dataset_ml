import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

beed_file_path = "BEED_Data.csv"
beed_data = pd.read_csv(beed_file_path)

X = beed_data.iloc[:, :16]
y = beed_data.iloc[:, 16]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0, train_size=0.8, stratify=y)

model = RandomForestClassifier(n_estimators=10, random_state=1)
model.fit(train_X, train_y)

predictions = model.predict(val_X)

accuracy = accuracy_score(val_y, predictions)
print(f"Accuracy: {accuracy:.3f}")
print(classification_report(val_y, predictions))
print(conf_matr := confusion_matrix(val_y, predictions))