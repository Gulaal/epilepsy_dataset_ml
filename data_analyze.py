import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def compare_signals():
    beed_data = pd.read_csv('BEED_Data.csv')
    
    first_col_0 = beed_data.loc[beed_data.iloc[:, -1] == 0, beed_data.columns[0]]
    first_col_1 = beed_data.loc[beed_data.iloc[:, -1] == 1, beed_data.columns[0]]
    first_col_2 = beed_data.loc[beed_data.iloc[:, -1] == 2, beed_data.columns[0]]
    first_col_3 = beed_data.loc[beed_data.iloc[:, -1] == 3, beed_data.columns[0]]
    
    second_col_0 = beed_data.loc[beed_data.iloc[:, -1] == 0, beed_data.columns[0]]
    second_col_1 = beed_data.loc[beed_data.iloc[:, -1] == 1, beed_data.columns[0]]
    second_col_2 = beed_data.loc[beed_data.iloc[:, -1] == 2, beed_data.columns[0]]
    second_col_3 = beed_data.loc[beed_data.iloc[:, -1] == 3, beed_data.columns[0]]

    labels = beed_data.iloc[:,-1]
    # plt.scatter(first_col_0, second_col_0, color='green')
    # plt.scatter(first_col_1, second_col_1, color='yellow')
    # plt.scatter(first_col_2, second_col_2, color='orange')
    # plt.scatter(first_col_3, second_col_3, color='red')


    plt.scatter([0 for _ in range(2000)], first_col_0, color='green')
    plt.scatter([1 for _ in range(2000, 4000)], first_col_1, color='yellow')
    plt.scatter([2 for _ in range(4000, 6000)], first_col_2, color='orange')
    plt.scatter([3 for _ in range(6000, 8000)], first_col_2, color='red')
    plt.show()

def compare_sums():

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    beed_data = pd.read_csv('BEED_Data.csv')

    beed_data["2SUM"] = beed_data.iloc[:,:-1].abs().sum(axis=1)
    print(beed_data.head())

    arr0 = np.array(beed_data.loc[beed_data.iloc[:, -2] == 0, "2SUM"])
    print(arr0)
    arr1 = np.array(beed_data.loc[beed_data.iloc[:, -2] == 1, "2SUM"])
    arr2 = np.array(beed_data.loc[beed_data.iloc[:, -2] == 2, "2SUM"])
    arr3 = np.array(beed_data.loc[beed_data.iloc[:, -2] == 3, "2SUM"])

    x_ax = [i for i in range(len(arr0))]

    print(arr1)

    axs[0, 0].bar(x_ax, arr0)
    axs[0, 1].bar(x_ax, arr1)
    axs[1, 0].bar(x_ax, arr2)
    axs[1, 1].bar(x_ax, arr3)

    plt.tight_layout()
    plt.show()

def compare_scaled_sums():
    scaler = StandardScaler()

    beed_data = pd.read_csv('BEED_Data.csv')
    columns = beed_data.columns
    beed_data_scaled = scaler.fit_transform(beed_data.iloc[:,:-1])
    
    df_scaled = pd.DataFrame(beed_data_scaled)
    df_scaled.columns = columns[:-1]
    df_scaled['y'] = beed_data['y']
    df_scaled["SUM"] = df_scaled.iloc[:,:-1].abs().sum(axis=1)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    arr0 = np.array(df_scaled.loc[df_scaled.iloc[:, -2] == 0, "SUM"])
    arr1 = np.array(df_scaled.loc[df_scaled.iloc[:, -2] == 1, "SUM"])
    arr2 = np.array(df_scaled.loc[df_scaled.iloc[:, -2] == 2, "SUM"])
    arr3 = np.array(df_scaled.loc[df_scaled.iloc[:, -2] == 3, "SUM"])
    x_ax = [i for i in range(len(arr0))]

    print(arr1)

    axs[0, 0].bar(x_ax, arr0)
    axs[0, 0].set_title(f"Class 0\nAverage sum {arr0.mean()}")
    axs[0, 1].bar(x_ax, arr1)
    axs[0, 1].set_title(f"Class 1\nAverage sum {arr1.mean()}")
    axs[1, 0].bar(x_ax, arr2)
    axs[1, 0].set_title(f"Class 2\nAverage sum {arr2.mean()}")
    axs[1, 1].bar(x_ax, arr3)
    axs[1, 1].set_title(f"Class 3\nAverage sum {arr3.mean()}")

    plt.tight_layout()
    plt.show()
    

if __name__ == '__main__':
    compare_scaled_sums()

    

    

    
    