import torch
from torch import nn
from util.update import LocalUpdate

'''
   inherit LocalUpdate class
'''
class LocalDevice(LocalUpdate):

    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))

        if args.gpu == 0:
            self.device = 'cuda'
        elif args.gpu == 1:
            self.device = 'mps'
        else:
            self.device = 'cpu'

        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)