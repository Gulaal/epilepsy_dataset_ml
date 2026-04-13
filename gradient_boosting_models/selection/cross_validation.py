import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import uniform, randint

beed_file_path = 'BEED_Data.csv'
beed_data = pd.read_csv(beed_file_path)

X = beed_data.iloc[:,:-1]
y = beed_data.iloc[:,-1]

model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.3, n_iter_no_change=20, random_state=1, max_depth=8, subsample=0.8)
skfold = StratifiedKFold(n_splits=8, shuffle=True, random_state=1)
scores = cross_validate(model, X, y, cv=skfold, scoring='accuracy', n_jobs=6)

print(scores)