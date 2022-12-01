import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def ctime_iter_plot(efl_ctime, dtfl_ctime):
    '''
    :return: present communication time-iter plot
    '''

    plt.figure()
    plt.plot(range(len(dtfl_ctime)), dtfl_ctime, label='Traditional FL')
    plt.plot(range(len(efl_ctime)), efl_ctime, label='eFL')
    plt.ylabel('Communication time(s)', fontsize=20)
    plt.xlabel('Iterations', fontsize=20)
    plt.legend()
    plt.show()


if __name__=='__main__':
    efl_df = pd.read_csv('../results/efl_dirty70_mlp.csv')
    dtfl_df = pd.read_csv('../results/tfl_dirty70_mlp.csv')
    efl_ctime = efl_df['com_time'].to_list()
    dtfl_ctime = dtfl_df['com_time'].to_list()
    ctime_iter_plot(efl_ctime, dtfl_ctime)