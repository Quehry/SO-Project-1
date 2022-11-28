from script.datasets import E3Datasets
from script.model import E3Model
from script.trainer import train_loop, test_loop
from torch.utils.data import DataLoader
import torch
from torch import nn

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_data = E3Datasets(num=7000, type='train', device=device)
    test_data = E3Datasets(num=3000, type='test', device=device)

    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=16, shuffle=False)
    
    model = E3Model().to(device)

    lr = 1e-5
    epoch = 1000

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for i in range(epoch):
        print(f"Epoch {i+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)