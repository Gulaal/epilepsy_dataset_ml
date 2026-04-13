import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

beed_file_path = 'BEED_Data.csv'
beed_data = pd.read_csv(beed_file_path)

X = beed_data.iloc[:,:-1]
y = beed_data.iloc[:,-1]

def train_and_validate(folds: KFold):
    global X, y, results
    max_accuracy = 0
    avg_accuracy = 0
    sum_accs = 0
    print(folds)
    for i, (train_index, test_index) in enumerate(folds.split(X, y)):
        print(f"Fold {i+1}")

        train_X, train_y, val_X, val_y = X.iloc[train_index], y.iloc[train_index], X.iloc[test_index], y.iloc[test_index]

        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=1, n_iter_no_change=10)
        model.fit(train_X, train_y)

        predictions = model.predict(val_X)
        accuracy = accuracy_score(val_y, predictions)
        matrix = confusion_matrix(val_y, predictions)

        max_accuracy = max(max_accuracy, accuracy)
        sum_accs += accuracy

        print(f"Accuracy: {accuracy}")
        print("=========================")
        print(f"Confusion matrix: \n {matrix}\n")
        print("=========================")

    avg_accuracy = sum_accs / folds.get_n_splits()

    print(f"Max accuracy: {max_accuracy}")    
    print(f"Average accuracy: {avg_accuracy}\n")

    return [folds, max_accuracy, avg_accuracy]

def stratified_kfold_selection():
    print("STRATIFIED")
    kf = StratifiedKFold(n_splits=5)
    return train_and_validate(kf)

def stratified_kfold_selection_shuffle():
    print("SHUFFLE")
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    return train_and_validate(kf)

def kfold_selection_shuffle():
    print("SHUFFLE")
    kf = KFold(n_splits=5, shuffle=True)
    return train_and_validate(kf)

def kfold_selection():
    print("WITHOUT SHUFFLE")
    kf = KFold(n_splits=5)
    return train_and_validate(kf)

def get_info():
    results = [
        kfold_selection_shuffle(),
        kfold_selection(),
        stratified_kfold_selection(),
        stratified_kfold_selection_shuffle(),
    ]
    return results

print(get_info())