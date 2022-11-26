import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def loss_iter_plot(tfl_loss_list, dtfl_loss_list):
    '''
    :return: present loss-iter plot
    '''

    plt.figure()
    plt.plot(range(len(dtfl_loss_list)), dtfl_loss_list, label='Traditional FL')
    plt.plot(range(len(tfl_loss_list)), tfl_loss_list, label='FL without dirty labels')
    plt.ylabel('Training Loss')
    plt.xlabel('Iterations')
    plt.grid()
    plt.legend()
    plt.show()


if __name__=='__main__':
    tfl_df = pd.read_csv('../results/tfl_nodirt_mlp.csv')
    dtfl_df = pd.read_csv('../results/tfl_dirty70_mlp.csv')
    tfl_loss_list = tfl_df['train_loss'].to_list()
    dtfl_loss_list = dtfl_df['train_loss'].to_list()
    loss_iter_plot(tfl_loss_list, dtfl_loss_list)