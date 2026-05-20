import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np
import time

beed_file_path = 'BEED_Data.csv'
beed_data = pd.read_csv(beed_file_path)

X = beed_data.iloc[:, :-1]
y = beed_data.iloc[:, -1]

def preprocess_data():
    global X, y
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y
    )
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=1)
    return sss, X_temp, X_test, y_temp, y_test

def time_decorator(func):
    def wrapper(*args):
        start_time = time.perf_counter()
        func(*args)
        print(time.perf_counter() - start_time)
    return wrapper


@time_decorator
def random_forest_train(sss, X_temp, X_test, y_temp, y_test):
    model = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=1, min_samples_split=2, random_state=1)
    
    scores = cross_validate(model, X_temp, y_temp, cv=sss, n_jobs=6, scoring='accuracy')
    np_scores = np.array(scores['test_score'])
    print("Random Forest (на train+val):")
    print(f"Средняя точность: {np_scores.mean():.4f}")
    print(f"Макс: {np_scores.max():.4f}, мин: {np_scores.min():.4f}")
    
    model.fit(X_temp, y_temp)
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(confusion_matrix(y_test, y_pred))
    print(f"Точность на тестовой выборке: {test_acc:.4f}\n")

@time_decorator
def gradient_boosting_train(sss, X_temp, X_test, y_temp, y_test):
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

if __name__ == "__main__":
    sss, X_temp, X_test, y_temp, y_test = preprocess_data()
    random_forest_train(sss, X_temp, X_test, y_temp, y_test)
    gradient_boosting_train(sss, X_temp, X_test, y_temp, y_test)