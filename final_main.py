import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np

beed_file_path = 'BEED_Data.csv'
beed_data = pd.read_csv(beed_file_path)

X = beed_data.iloc[:, :-1]
y = beed_data.iloc[:, -1]

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def random_forest_train():
    global X_temp, y_temp
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=1)
    model = RandomForestClassifier(n_estimators=100, random_state=1)
    
    scores = cross_validate(model, X_temp, y_temp, cv=sss, n_jobs=6, scoring='accuracy')
    np_scores = np.array(scores['test_score'])
    print("Random Forest (на train+val):")
    print(f"Средняя точность: {np_scores.mean():.4f}")
    print(f"Макс: {np_scores.max():.4f}, мин: {np_scores.min():.4f}")
    
    model.fit(X_temp, y_temp)
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Точность на тестовой выборке: {test_acc:.4f}\n")

def gradient_boosting_train():
    global X_temp, y_temp
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=1)
    model = XGBClassifier(n_estimators=100, learning_rate=0.3, max_depth=8, subsample=0.8, random_state=1)
    
    scores = cross_validate(model, X_temp, y_temp, cv=sss, n_jobs=6, scoring='accuracy')
    np_scores = np.array(scores['test_score'])
    print("XGBoost (на train+val):")
    print(f"Средняя точность: {np_scores.mean():.4f}")
    print(f"Макс: {np_scores.max():.4f}, мин: {np_scores.min():.4f}")
    
    model.fit(X_temp, y_temp)
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(confusion_matrix(y_test, y_pred))
    print(f"Точность на тестовой выборке: {test_acc:.4f}\n")

random_forest_train()
gradient_boosting_train()