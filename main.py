from scripts.datasets import E3Datasets
from scripts.model import E3Model
from scripts.trainer import train_loop, test_loop
from torch.utils.data import DataLoader
import torch
from torch import nn

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets_num = 10000
    batch_size = 16
    lr = 1e-5
    epoch = 5000
    seed = 505
    train = True

    # create train and test data for e3 task
    train_data = E3Datasets(num=int(datasets_num*0.7), type='train', device=device, seed=seed)
    test_data = E3Datasets(num=int(datasets_num*0.3), type='test', device=device, seed=seed)

    # create dataloader
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # create e3model
    model = E3Model(seed=seed).to(device)

    # create loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # train
    if train:
        for i in range(epoch):
            print(f"Epoch {i+1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer)
            test_loop(test_dataloader, model, loss_fn)
        # torch.save(model.state_dict(), 'model_weights.pth')

    # eval
    if not train:
        model.load_state_dict(torch.load('model_weights.pth'))
        model.eval()

        input = torch.tensor([1,-1,-1,1,1.5,0.1], dtype=torch.float).reshape(3,2).to(device=device)
        print(input)
        print(model(input).argmax(dim=1))
