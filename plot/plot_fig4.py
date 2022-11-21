import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def cdf(sim_list, model_name):

    sim_list.sort()

    sim_mean = np.mean(sim_list)
    sim_std = np.std(sim_list)

    sim_pdf = norm.pdf(sim_list, sim_mean, sim_std)

    X = sim_list
    Y = np.cumsum(sim_pdf)
    print(Y)

    plt.figure()
    plt.xlabel('Similarity between model parameters', fontsize=30)
    plt.ylabel('CDF', fontsize=30)
    plt.plot(X, Y, label=model_name)
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('../sim.csv')
    print(df)
