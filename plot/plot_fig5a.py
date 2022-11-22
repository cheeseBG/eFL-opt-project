import matplotlib.pyplot as plt


def acc_iter_plot(acc_list):
    '''

    :param acc_list: [[fl_type_name1, list of acc], [fl_type_name2, list of acc], ...]
    :return: present acc-iter plot
    '''

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
