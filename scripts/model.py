from torch import nn
import torch

class E3Model(nn.Module):
    def __init__(self, seed=47):
        super(E3Model, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )
        self.seed = seed
        self.set_seed()
        self.reset_parameters()
        
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
    def set_seed(self):
        torch.manual_seed(self.seed)

    def reset_parameters(self):
        for layer in self.linear_relu_stack:
            if type(layer) == nn.Linear:
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.normal_(layer.bias, std=0.01)

if __name__ == "__main__":
    model = E3Model()
    print(model.linear_relu_stack[2].bias.data)