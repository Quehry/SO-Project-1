from scripts.datasets import E3Datasets
from scripts.model import E3Model
from scripts.trainer import train_epoch
from scripts.animator import Animator
from scripts.test_sgd import test_sgd
from scripts.test_batch_size import test_batch_size
from scripts.test_initialisation import test_initialisation
from scripts.visualisation import visualisation_results
from torch.utils.data import DataLoader
import torch
from torch import nn

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets_num = 100000
    batch_size = 32
    lr = 1e-5
    num_epochs = 100
    seed = 505
    train = True
    test_SGD = False
    bool_test_batch_size = False
    bool_test_initialisation = False

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
        if test_SGD:
            lr_test_loss, lr_test_acc = test_sgd(train_dataloader, test_dataloader, model, loss_fn, num_epochs)

        if bool_test_batch_size:
            lr_test_loss, lr_test_acc = test_batch_size(train_data, test_data, model, loss_fn, optimizer, num_epochs)

        if bool_test_initialisation:
            lr_test_loss, lr_test_acc = test_initialisation(train_dataloader, test_dataloader, model, loss_fn, optimizer, num_epochs)

        
        # create accumulator and animator
        # animator = Animator(xlabel="epoch", xlim=[1,num_epochs],legend=["train_loss", "test_loss", "train_acc", "test_acc"],nrows=2,ncols=1, figsize=[14, 10])
        # train_epoch(train_dataloader, test_dataloader, model, loss_fn, optimizer, num_epochs, animator)
        # torch.save(model.state_dict(), '/root/autodl-tmp/models/base.pth')

    # eval
    if not train:
        model.load_state_dict(torch.load('model_weights.pth'))
        model.eval()

        visualisation_results(test_dataloader, model) #TODO

        input = torch.tensor([1,-1,-1,1,1.5,0.1], dtype=torch.float).reshape(3,2).to(device=device)
        print(input)
        print(model(input).argmax(dim=1))
