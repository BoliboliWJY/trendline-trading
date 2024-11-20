import matplotlib.pyplot as plt

def plot_data(data):
    plt.plot(data[:, 0], data[:, 1], label='Low')
    plt.plot(data[:, 0], data[:, 2], label='High')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
