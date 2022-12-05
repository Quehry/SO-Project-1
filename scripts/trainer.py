# -*-coding:utf-8 -*-
"""
Chinese Name: Que Haoran/Song Zhenghao/Cai Zhuojiang/Ji Yuwen
French Name: Francis/Herve/Evan/Neo
Student Number: SY2224124/ZY2224114/ZY2224102/ZY2224109
Date: 2022/12/3
"""

import torch

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batch = len(dataloader)
    correct = 0
    loss_t = 0
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_t += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss_t / num_batch, correct / size


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # pred.argmax(1) equals softmax(pred).argmax(1)

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct


def train_epoch(train_dataloader, test_dataloader, model, loss_fn, optimizer, num_epochs, animator):
    for i in range(num_epochs):
            print(f"Epoch {i+1}\n-------------------------------")
            train_loss, train_acc = train_loop(train_dataloader, model, loss_fn, optimizer)
            test_loss, test_acc = test_loop(test_dataloader, model, loss_fn)
            metrics = (train_loss, test_loss, train_acc, test_acc,)
            animator.add(i+1, metrics)