from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from read_info import read_info

def preprocess_data():

    X, y = read_info()
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y
    )
    return X_temp, X_test, y_temp, y_test

def get_parameters(model, param_grid):

    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=6,
        verbose=1
    )

    X_temp, X_test, y_temp, y_test = preprocess_data()

    gs.fit(X_temp, y_temp)
    
    final_model = gs.best_estimator_
    best_params = gs.best_params_

    print(best_params)

    y_pred = final_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Точность лучшей модели на тестовых данных: {acc}")
    return final_model, best_params


rf_param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

xgb_param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [6, 8, 12, 16],
    'learning_rate':[0.1, 0.2, 0.3, 0.4],
    'subsample': [0.7, 0.8, 0.9]
}

rf = get_parameters(RandomForestClassifier(random_state=1), rf_param_grid)
xgb = xgb_param_grid(XGBClassifier(random_state=1), xgb_param_grid)