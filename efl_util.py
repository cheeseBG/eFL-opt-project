import numpy as np


def cos_similarity(keys, l_params, g_params):
    '''
    :param l_params: Local model parameters (i th iteration)
    :param g_params: Global model parameters (i-1 th iteration)
    :return: similarity value
    '''
    sim_list = list()


    for key in keys:

        l_vec = l_params[key]
        g_vec = g_params[key]
        print(key)
        print(l_vec.shape)

        l_norm = np.linalg.norm(l_vec, 2)
        g_norm = np.linalg.norm(g_vec, 2)
        lg_dot = np.dot(np.transpose(l_vec), g_vec)

        sim = lg_dot / (l_norm * g_norm)
        sim_list.append([key, sim])

    return sim_list
