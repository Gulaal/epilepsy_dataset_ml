import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score

beed_file_path = 'BEED_Data.csv'
beed_data = pd.read_csv(beed_file_path)

X = beed_data.iloc[:,:-1]
y = beed_data.iloc[:,-1]

skfold = StratifiedKFold(n_splits=8, shuffle=True, random_state=1)

model = XGBClassifier(n_estimators = 500, learning_rate = 0.2, eval_metric='error', max_depth=8, random_state=1, subsample=0.8)

scores = cross_validate(model, X, y, cv=skfold, n_jobs=6, scoring='accuracy')

print(scores)