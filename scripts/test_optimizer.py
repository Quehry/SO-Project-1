# -*-coding:utf-8 -*-
"""
Chinese Name: Que Haoran/Song Zhenghao/Cai Zhuojiang/Ji Yuwen
French Name: Francis/Herve/Evan/Neo
Student Number: SY2224124/ZY2224114/ZY2224102/ZY2224109
Date: 2022/12/3
"""

import torch
from scripts.trainer import train_loop, test_loop
from matplotlib import pyplot as plt
from IPython import display

def test_optimizer(train_dataloader, test_dataloader, model, loss_fn, num_epochs):
    lr_test_loss = [[]for _ in range(2)]
    lr_test_acc = [[]for _ in range(2)]
    x = [[i+1 for i in range(num_epochs)] for _ in range(2)]

    model.reset_parameters(init=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    for _ in range(num_epochs):
        train_loss, train_acc = train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loss, test_acc = test_loop(test_dataloader, model, loss_fn)
        lr_test_loss[0].append(test_loss)
        lr_test_acc[0].append(test_acc)

    model.reset_parameters(init=0)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-5, rho=0.9, eps=1e-6, weight_decay=0)
    for _ in range(num_epochs):
        train_loss, train_acc = train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loss, test_acc = test_loop(test_dataloader, model, loss_fn)
        lr_test_loss[1].append(test_loss)
        lr_test_acc[1].append(test_acc)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=[14,10])

    for x_,y_ in zip(x,lr_test_loss):
        axes[0].plot(x_, y_)
    
    for x_,y_ in zip(x,lr_test_acc):
        axes[1].plot(x_, y_)

    axes[0].set_xlabel('epoch')
    axes[1].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[1].set_ylabel('accuracy')
    axes[0].set_xscale('linear')
    axes[0].set_yscale('linear')
    axes[1].set_xscale('linear')
    axes[1].set_yscale('linear')
    axes[0].set_xlim([1,num_epochs])
    axes[0].set_ylim([0,1.5])
    axes[1].set_xlim([1,num_epochs])
    axes[1].set_ylim([0,1])
    axes[0].legend(["SGD","Adadelta"], loc=1)
    axes[1].legend(["SGD","Adadelta"], loc=3)
    axes[0].grid()
    axes[1].grid()

    display.display(fig)
    display.clear_output(wait=True)

    return lr_test_loss, lr_test_acc
    
