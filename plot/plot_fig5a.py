import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def acc_iter_plot(tfl_acc_list, dtfl_acc_list, efl_acc_list):
    '''

    :param acc_list: [[fl_type_name1, list of acc], [fl_type_name2, list of acc], ...]
    :return: present acc-iter plot
    '''

    plt.figure()
    plt.plot(range(len(dtfl_acc_list)), dtfl_acc_list, label='Traditional FL')
    plt.plot(range(len(tfl_acc_list)), tfl_acc_list, label='FL without dirty labels')
    plt.plot(range(len(efl_acc_list)), efl_acc_list, label='eFL')
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.legend()
    plt.grid()
    plt.ylim([0.3, 1.0])
    plt.show()


if __name__=='__main__':
    tfl_df = pd.read_csv('../results/tfl_nodirt_mlp.csv')
    dtfl_df = pd.read_csv('../results/tfl_dirty70_mlp.csv')
    efl_df = pd.read_csv('../results/efl_dirty70_mlp.csv')

    tfl_acc_list = tfl_df['test_acc'].to_list()
    dtfl_acc_list = dtfl_df['test_acc'].to_list()
    efl_acc_list = efl_df['test_acc'].to_list()
    acc_iter_plot(tfl_acc_list, dtfl_acc_list, efl_acc_list)