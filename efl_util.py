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

        l_norm = np.linalg.norm(l_vec)
        g_norm = np.linalg.norm(g_vec)
        lg_dot = sum(np.dot(np.transpose(l_vec[i]), g_vec[i]) for i in range(0, len(l_vec)))

        sim = lg_dot / float(l_norm * g_norm)
        sim_list.append([key, sim])

    # Return average cos_sim
    avg_sim = sum(s[1] for s in sim_list) / len(sim_list)

    return avg_sim

