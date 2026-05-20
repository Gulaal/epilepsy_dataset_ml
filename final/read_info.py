import pandas as pd

def read_info():
    data = pd.read_csv('BEED_Data.csv')
    X, y = data.iloc[:,:-1], data.iloc[:,-1]
    return X, y