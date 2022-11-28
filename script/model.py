import torch
from torch import nn
from torch.nn.functional import one_hot

class E3Model(nn.Module):
    def __init__(self):
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
        # self.softmax = nn.Softmax(dim=1)

        
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        # pred_probab = self.softmax(logits)
        # y_pred = pred_probab.argmax(dim=1)
        # y_pred = one_hot(y_pred, num_classes=3)
        return logits
    