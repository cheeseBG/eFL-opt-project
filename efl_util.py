import numpy as np


def cos_similarity(args, keys, l_params, g_params):
    '''
    :param1 keys: name list of model layers
    :param2 l_params: Local model parameters (i th iteration)
    :param3 g_params: Global model parameters (i-1 th iteration)
    :return: similarity value
    '''
    sim_list = list()

    for key in keys:

        l_vec = l_params[key].to('cpu')
        g_vec = g_params[key].to('cpu')
        l_norm = np.linalg.norm(l_vec)
        g_norm = np.linalg.norm(g_vec)

        if args.model == 'cnn':
            lg_dot = sum(np.dot(np.transpose(l_vec[i].reshape(-1,1)), g_vec[i].reshape(-1,1)) for i in range(0, len(l_vec)))
        else:
            lg_dot = sum(np.dot(np.transpose(l_vec[i]), g_vec[i]) for i in range(0, len(l_vec)))

        sim = float(lg_dot / float(l_norm * g_norm))
        sim_list.append([key, sim])

    # Return average cos_sim
    avg_sim = sum(s[1] for s in sim_list) / len(sim_list)

    # Normalize range of [0,1]
    norm_sim = avg_sim * 0.5 + 0.5

    return norm_sim

