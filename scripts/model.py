# -*-coding:utf-8 -*-
"""
Chinese Name: Que Haoran/Song Zhenghao/Cai Zhuojiang/Ji Yuwen
French Name: Francis/Herve/Evan/Neo
Student Number: SY2224124/ZY2224114/ZY2224102/ZY2224109
Date: 2022/12/3
"""

from torch import nn
import torch
import math

class E3Model(nn.Module):
    def __init__(self, seed=47):
        super(E3Model, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )
        self.seed = seed
        self.set_seed()
        self.reset_parameters(2)
        
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
    def set_seed(self):
        torch.manual_seed(self.seed)

    def reset_parameters(self, init):
        for layer in self.linear_relu_stack:
            if type(layer) == nn.Linear:

                if init==0:
                    nn.init.kaiming_normal_(layer.weight, a=math.sqrt(5))
                    if layer.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(layer.bias, -bound, bound)
                
                if init==1:
                    nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('relu'))
                    if layer.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(layer.bias, -bound, bound)

                if init==2:
                    nn.init.normal_(layer.weight, std=0.1)
                    if layer.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(layer.bias, -bound, bound)
                
if __name__ == "__main__":
    model = E3Model()
    print(model.linear_relu_stack[2].weight.data)