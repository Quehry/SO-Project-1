# -*-coding:utf-8 -*-
"""
Chinese Name: Que Haoran/Song Zhenghao/Cai Zhuojiang/Ji Yuwen
French Name: Francis/Herve/Evan/Neo
Student Number: SY2224124/ZY2224114/ZY2224102/ZY2224109
Date: 2022/12/3
"""

import torch
from matplotlib import pyplot as plt

def visualisation_results(test_dataloader, model):
    correct_X = torch.tensor([0,0],dtype=torch.float, device='cuda').reshape(1,2)
    uncorrect_X = torch.tensor([0,0],dtype=torch.float, device='cuda').reshape(1,2)
    for X,y in test_dataloader:
        pred = model(X)
        correct = pred.argmax(dim=1)==y
        false = pred.argmax(dim=1)!=y  
        tmp = X[correct]
        tmp_f = X[false]
        correct_X = torch.concat([correct_X, tmp], dim=0)
        uncorrect_X = torch.concat([uncorrect_X, tmp_f], dim=0)
    
    correct_X = correct_X[1:]
    uncorrect_X = uncorrect_X[1:]
    print(correct_X)
    print(correct_X.shape)
    print(uncorrect_X)
    print(uncorrect_X.shape)

    correct_Xx = correct_X[:,0]
    correct_Xy = correct_X[:,1]

    uncorrect_Xx = uncorrect_X[:,0]
    uncorrect_Xy = uncorrect_X[:,1]

    plt.figure(figsize=[14,10])
    plt.scatter(correct_Xx.cpu(), correct_Xy.cpu(), s=10, color='blue')
    plt.scatter(uncorrect_Xx.cpu(), uncorrect_Xy.cpu(), s=10, color='red')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(["correct", "uncorrect"])

    plt.show()