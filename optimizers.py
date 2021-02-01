from train_script import train
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, Adafactor
from datetime import datetime
import torch.optim as optim


### HYPERPARAMETERS left to implement###
local_attention_window = [] #TODO
n_list = [3, 5, 7]   # number of top predictions to take into account.

def tune_optimizers(model, train_loader, val_loader, optimizer_type='adafactor',
                    lr_list=[1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-5, 1e-6, 5e-6],
                    eps_list = [1e-07, 1e-08, 1e-09],
                    momentum_list=[0.7, 0.9], num_epochs=3):

    ##### ADAMW #####
    if optimizer_type=='adamw':
        for lr in lr_list:
            optimizer = AdamW(model.parameters(), lr=lr, relative_step=False)
            summary_path = f'runs/longformer_{len(train_loader)}samples_{num_epochs}epochs_{lr}lr'
            writer = SummaryWriter(summary_path)
            train(train_loader, val_loader, optim=optimizer, writer=writer, num_epochs=num_epochs)

    ##### ADAFACTOR ######
    if optimizer_type=='adafactor':
        for lr in lr_list:
            optimizer = Adafactor(model.parameters(), lr=lr, relative_step=False)
            summary_path = f'runs/longformer_{len(train_loader)}samples_{num_epochs}epochs_{lr}lr'
            writer = SummaryWriter(summary_path)
            train(train_loader, val_loader, optim=optimizer, writer=writer, num_epochs=num_epochs)
        optimizer = Adafactor(model.parameters(), warmup_init=True)
        summary_path = f'runs/longformer_{len(train_loader)}samples_{num_epochs}epochs_WARMUP'
        writer = SummaryWriter(summary_path)
        train(train_loader, val_loader, optim=optimizer, writer=writer, num_epochs=num_epochs)

    ##### SGD #####
    if optimizer_type=='sgd':
        for lr in lr_list:
            for momentum in momentum_list:
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
                summary_path = f'runs/longformer_{len(train_loader)}samples_{num_epochs}epochs_{lr}lr_{momentum}momentum'
                writer = SummaryWriter(summary_path)
                train(train_loader, val_loader, optim=optimizer, writer=writer, num_epochs=num_epochs)

    ##### RMSProp #####
    if optimizer_type=='rmsprop':
        for lr in lr_list:
            for eps in eps_list:
                optimizer = optim.RMSprop(model.parameters(), lr=lr, eps=eps, centered=False)
                summary_path = f'runs/longformer_{len(train_loader)}samples_{num_epochs}epochs_{lr}lr_{eps}eps'
                writer = SummaryWriter(summary_path)
                train(train_loader, val_loader, optim=optimizer, writer=writer, num_epochs=num_epochs)


