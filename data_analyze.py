import matplotlib.pyplot as plt
import pandas as pd

colors = {
    0: 'green',
    1: 'yellow',
    2: 'orange',
    3: 'red',
}

if __name__ == '__main__':
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

    

    

    
    