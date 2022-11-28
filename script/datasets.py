from torch.utils.data import Dataset
import torch


class E3Datasets(Dataset):
    def __init__(self, num=1000, type='train', device='cuda'):
        self.num = num
        self.type = type
        self.device = device
        self.feature, self.label = self.create_data()


    def create_data(self):
        x = torch.rand([self.num, 2], dtype=torch.float) * 3
        p = torch.rand([self.num, 2]) > 0.5
        x[p] = x[p] *-1
        e1 = torch.abs(x[:,0]-1.5)+torch.abs(x[:,1])< 1
        e2 = torch.sqrt(torch.pow(x[:,0]+1,2)+torch.pow(x[:,1]-1,2)) < 1
        y = torch.zeros(self.num, dtype=torch.long)
        for index in range(self.num):
            if e1[index]:
                y[index]=0
            elif e2[index]:
                y[index]=1
            else:
                y[index]=2
        if self.device == 'cuda':
            x = x.to(self.device)
            y = y.to(self.device, dtype=torch.long)
        return x,y

    def __len__(self):
        return self.num
    
    def __getitem__(self, index):
        return self.feature[index], self.label[index]




if __name__ == "__main__":
    dataset = E3Datasets()
    print(dataset.feature[50:100])
    print(dataset.label[50:100])