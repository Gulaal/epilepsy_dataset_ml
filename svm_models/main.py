from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import pandas as pd

beed_file_path = "./BEED_Data.csv"
beed_data = pd.read_csv(
    beed_file_path
)

X, y = beed_data.iloc[:, :-1], beed_data.iloc[:, -1]

skf = StratifiedKFold(5, shuffle=True, random_state=1)

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"Fold {i}")

    model = SVC(probability=True, random_state=1)
    model.fit(X.iloc[train_index], y.iloc[train_index])
    predictions = model.predict(X.iloc[test_index])
    accuracy = accuracy_score(y.iloc[test_index], predictions)
    print(f"Accuracy per fold {accuracy}")

    