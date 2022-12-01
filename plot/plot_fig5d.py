import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def cnn_acc_iter_plot(tfl_acc_list, dtfl_acc_list, efl_acc_list):
    '''

    :param acc_list: [[fl_type_name1, list of acc], [fl_type_name2, list of acc], ...]
    :return: present acc-iter plot
    '''

    plt.figure()
    plt.plot(range(len(dtfl_acc_list)), dtfl_acc_list, label='Traditional FL')
    plt.plot(range(len(tfl_acc_list)), tfl_acc_list, label='FL without dirty labels')
    plt.plot(range(len(efl_acc_list)), efl_acc_list, label='eFL')
    plt.ylabel('Accuracy', fontsize=20)
    plt.xlabel('Iterations', fontsize=20)
    plt.legend()
    plt.grid()
    plt.ylim([0.0, 1.0])
    plt.show()


if __name__=='__main__':
    tfl_df = pd.read_csv('../results/tfl_nodirt_cnn.csv')
    dtfl_df = pd.read_csv('../results/tfl_dirty70_cnn.csv')
    efl_df = pd.read_csv('../results/efl_dirty70_cnn.csv')

    tfl_acc_list = tfl_df['train_acc'].to_list()
    dtfl_acc_list = dtfl_df['train_acc'].to_list()
    efl_acc_list = efl_df['train_acc'].to_list()
    cnn_acc_iter_plot(tfl_acc_list, dtfl_acc_list, efl_acc_list)