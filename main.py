from scripts.datasets import E3Datasets
from scripts.model import E3Model
from scripts.trainer import train_loop, test_loop
from torch.utils.data import DataLoader
import torch
from torch import nn

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16
    lr = 1e-5
    epoch = 1000
    seed = 42

    # create train and test data for e3 task
    train_data = E3Datasets(num=7000, type='train', device=device, seed=seed)
    test_data = E3Datasets(num=3000, type='test', device=device, seed=seed)

    # create dataloader
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # create e3model
    model = E3Model().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for i in range(epoch):
        print(f"Epoch {i+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)