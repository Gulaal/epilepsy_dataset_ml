import matplotlib.pyplot as plt

def draw_loss(epochs, avg_loss):
    X = [i + 1 for i in range(epochs)]
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Average loss")
    plt.plot(X, avg_loss)
    plt.show()