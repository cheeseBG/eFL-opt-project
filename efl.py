#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from util.options import args_parser
from util.update import LocalUpdate, test_inference
from util.models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from util.utils import get_dataset, average_weights, exp_details

from efl_util import cos_similarity
from local_device import LocalDevice
from plot.plot_fig4 import cdf


if __name__ == '__main__':

    lan_dict = {
        1: 1/15,  # bandwidth
        2: 1/20,
        3: 1/60,
        4: 1/17
    }

    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    # Communication Time setting (Seconds)
    total_com_time = 0.0
    wan_bandwidth = 1 / 2  # M/(B*SNR)
    wan_com_time = len(lan_dict) * wan_bandwidth
    lan_com_time = wan_bandwidth
    com_time_list = []

    # dirty proportion: 1/5 propotion of end devices
    dirty_ed = int(args.num_users * (4 / 5))
    exception_list = []

    # Set similarity threshold
    thres = 0
    if args.model == 'mlp':
        thres = 0.98
    else:
        thres = 0.9999


    if args.gpu==0:
        print('\n### Use GPU ###\n')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
        device = 'cuda:0' if args.gpu == 0 else 'cpu'
    # For m1 GPU acc
    elif args.gpu==1:
        print('\n### Use mps ###\n')
        device = torch.device("mps:0" if torch.backends.mps.is_available() else "cpu")
        device = 'mps' if args.gpu==1 else 'cpu'
    else:
        print('Nope')
        device='mps'

    # load dataset and user groups
    # each user has own dataset
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural network
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer perceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, val_loss_list, net_list = [], [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    # cosine similarity
    cos_sim_list = []

    idxs_users = []
    for i in range(0, args.num_users):
        lan_class = np.random.randint(1, 5)
        idxs_users.append([i, lan_class])

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)  # fraction of clients (default: 10)

        except_users = 0

        for idx, lan in idxs_users:
            local_model = LocalDevice(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch, dirty_ed=dirty_ed)

            if dirty_ed > 0:
                dirty_ed -= 1

            # Calculate cosine similarity with global weight(t-1) and local weight(t)
            cos_sim = cos_similarity(args, global_weights.keys(), w, global_weights)
            cos_sim_list.append(cos_sim)

            # If similarity value is smaller than threshold, not aggregate
            if cos_sim < thres and epoch != 0:
                except_users += 1
                continue

            # Calculate LAN aggregation time
            total_com_time += lan_com_time * lan_dict[lan]

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))


        # Add 1epoch comunication time
        total_com_time += wan_com_time
        com_time_list.append(total_com_time)

        exception_list.append(except_users)

        if local_weights:
            # update global weights
            global_weights = average_weights(local_weights)

            # update global weights
            global_model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)
        else:
            print('## Except Users ##')
            print(except_users)
            train_loss.append(train_loss[-1])

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # Evaluation
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        val_loss_list.append(test_loss)
        val_acc_list.append(test_acc)

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    print("### cos ###")
    print(sum(cos_sim_list) / len(cos_sim_list))
    print('min: {} max: {}'.format(min(cos_sim_list), max(cos_sim_list)))
    ''' 
    Threshold for 
        - MLP {0.95, 0.955, 0.96, 0.965, 0.97, 0.975, *0.98, 0.985}
        - CNN {0.9995, 0.9996, 0.9997, 0.9998, *0.9999}
    '''

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    import pandas as pd

    total_results = {
        'train_acc': train_accuracy,
        'train_loss': train_loss,
        'test_acc': val_acc_list,
        'test_loss': val_loss_list,
        'com_time': com_time_list,
        'except_users': exception_list,

    }
    sim_result = {
        'cos_sim': cos_sim_list
    }

    df = pd.DataFrame(total_results)

    df.to_csv('results/efl_dirty{}_{}.csv'.format(str(args.dirty), args.model))

    sim_df = pd.DataFrame(sim_result)
    sim_df.to_csv('results/efl_dirty{}_{}_sim.csv'.format(str(args.dirty), args.model))


    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # # Saving the objects train_loss and train_accuracy:
    # file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)
    #
    # with open(file_name, 'wb') as f:
    #     pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
