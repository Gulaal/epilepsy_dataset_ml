import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

beed_file_path = 'BEED_Data.csv'
beed_data = pd.read_csv(beed_file_path)

X = beed_data.iloc[:,:-1]
y = beed_data.iloc[:,-1]