# -*-coding:utf-8 -*-
"""
Chinese Name: Que Haoran/Song Zhenghao/Cai Zhuojiang/Ji Yuwen
French Name: Francis/Herve/Evan/Neo
Student Number: SY2224124/ZY2224114/ZY2224102/ZY2224109
Date: 2022/12/3
"""

from scripts.trainer import train_loop, test_loop
from matplotlib import pyplot as plt
from IPython import display

def test_initialisation(train_dataloader, test_dataloader, model, loss_fn, optimizer, num_epochs):
    lr_test_loss = [[]for _ in range(3)]
    lr_test_acc = [[]for _ in range(3)]
    x = [[i+1 for i in range(num_epochs)] for _ in range(3)]
    init_list = [0,1,2]

    for i, init in enumerate(init_list):
        model.reset_parameters(init=init)
        for _ in range(num_epochs):
            train_loss, train_acc = train_loop(train_dataloader, model, loss_fn, optimizer)
            test_loss, test_acc = test_loop(test_dataloader, model, loss_fn)
            lr_test_loss[i].append(test_loss)
            lr_test_acc[i].append(test_acc)

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
    axes[0].legend(["kaiming_uniform_","xavier_uniform_","normal_"], loc=1)
    axes[1].legend(["kaiming_uniform_","xavier_uniform_","normal_"], loc=3)
    axes[0].grid()
    axes[1].grid()

    display.display(fig)
    display.clear_output(wait=True)

    return lr_test_loss, lr_test_acc
    
