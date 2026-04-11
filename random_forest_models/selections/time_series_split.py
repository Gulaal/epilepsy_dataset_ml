import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

beed_file_path = 'BEED_Data.csv'
beed_data = pd.read_csv(beed_file_path)

X = beed_data.iloc[:,:-1]
y = beed_data.iloc[:,-1]

def tss_train_and_validate():
    global X, y

    tss = TimeSeriesSplit(n_splits=5, test_size=1333)
    
    sum_accur = 0
    avg_accur = 0
    max_accur = 0

    for i, (train_index, val_index) in enumerate(tss.split(X)):
        print(f"Fold {i+1}")

        train_X, train_y, val_X, val_y = X.iloc[train_index], y.iloc[train_index], X.iloc[val_index], y.iloc[val_index]
        
        model = RandomForestClassifier(n_estimators=100, random_state=1)
        model.fit(train_X, train_y)
        predictions = model.predict(val_X)

        accuracy = accuracy_score(val_y, predictions)
        conf_matrix = confusion_matrix(val_y, predictions)

        sum_accur += accuracy
        max_accur = max(max_accur, accuracy)

        print(f"Accuracy: {accuracy}")
        print("=================================")
        print(conf_matrix)
        print("=================================\n")

    avg_accur = sum_accur / tss.get_n_splits()
    print(f"Average: {avg_accur}")
    print(f"Max: {max_accur}")

tss_train_and_validate()