import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def ctime_iter_plot(dtfl_ctime):
    '''
    :return: present communication time-iter plot
    '''

    plt.figure()
    plt.plot(range(len(dtfl_ctime)), dtfl_ctime, label='Traditional FL')
    #plt.plot(range(len(tfl_loss_list)), tfl_loss_list, label='FL without dirty labels')
    plt.ylabel('Communication time(s)')
    plt.xlabel('Iterations')
    plt.legend()
    plt.show()


if __name__=='__main__':
    dtfl_df = pd.read_csv('../results/efl_dirty70_mlp.csv')
    dtfl_ctime = dtfl_df['com_time'].to_list()
    ctime_iter_plot(dtfl_ctime)