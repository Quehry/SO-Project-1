from torch.utils.data import Dataset
import torch


class E3Datasets(Dataset):
    def __init__(self, num=1000, type='train', device='cuda', seed=42):
        self.num = num
        self.type = type
        self.device = device
        self.seed = seed
        self.set_seed()
        self.feature, self.label = self.create_data()


    # def create_data(self):
    #     x = torch.rand([self.num, 2], dtype=torch.float) * 3
    #     p = torch.rand([self.num, 2]) > 0.5
    #     x[p] = x[p] * -1
    #     e1 = torch.abs(x[:,0]-1.5) + torch.abs(x[:,1]) < 1
    #     e2 = torch.sqrt(torch.pow(x[:,0]+1,2) + torch.pow(x[:,1]-1,2)) < 1
    #     y = torch.zeros(self.num, dtype=torch.long)
    #     for index in range(self.num):
    #         if e1[index]:
    #             y[index]=0
    #         elif e2[index]:
    #             y[index]=1
    #         else:
    #             y[index]=2
    #     if self.device == 'cuda':
    #         x = x.to(self.device)
    #         y = y.to(self.device, dtype=torch.long)
    #     return x, y

    def create_data(self):
        self.divide_num = int(self.num / 3)
        x1 = torch.tensor([1.5,0],dtype=torch.float).reshape(1,2)
        while True:
            tmp = torch.rand([self.num, 2], dtype=torch.float) * 3
            p = torch.rand([self.num, 2]) > 0.5
            tmp[p] = tmp[p] * -1
            e1 = torch.abs(tmp[:,0]-1.5) + torch.abs(tmp[:,1]) < 1
            tmp = tmp[e1]
            x1 = torch.concat([x1,tmp], dim=0)
            if x1.shape[0] > self.divide_num:
                x1 = x1[:self.divide_num,:]
                break
        y1 = torch.zeros(x1.shape[0], dtype=torch.long)

        x2 = torch.tensor([-1,1],dtype=torch.float).reshape(1,2) 
        while True:
            tmp = torch.rand([self.num, 2], dtype=torch.float) * 3
            p = torch.rand([self.num, 2]) > 0.5
            tmp[p] = tmp[p] * -1
            e2 = torch.sqrt(torch.pow(tmp[:,0]+1,2) + torch.pow(tmp[:,1]-1,2)) < 1
            tmp = tmp[e2]
            x2 = torch.concat([x2,tmp], dim=0)
            if x2.shape[0] > self.divide_num:
                x2 = x2[:self.divide_num]
                break
        y2 = torch.ones(x2.shape[0], dtype=torch.long)

        x3 = torch.zeros([1,2], dtype=torch.float)
        while True:
            tmp = torch.rand([self.num, 2], dtype=torch.float) * 3
            p = torch.rand([self.num, 2]) > 0.5
            tmp[p] = tmp[p] * -1
            e3_1 = torch.abs(tmp[:,0]-1.5) + torch.abs(tmp[:,1]) >= 1 
            e3_2 = torch.sqrt(torch.pow(tmp[:,0]+1,2) + torch.pow(tmp[:,1]-1,2)) >= 1
            e3_list = []
            for index in range(tmp.shape[0]):
                if e3_1[index] and e3_2[index]:
                    e3_list.append(index)
            tmp = tmp[e3_list]
            x3 = torch.concat([x3,tmp], dim=0)
            if x3.shape[0] > self.divide_num:
                x3 = x3[:self.divide_num]
                break
        y3 = torch.ones(x3.shape[0], dtype=torch.long)
        y3 *= 2 

        x = torch.concat([x1,x2,x3],dim=0)
        y = torch.concat([y1,y2,y3],dim=0)
        if self.device == 'cuda':
            x = x.to(self.device)
            y = y.to(self.device, dtype=torch.long)
        return x, y

    def __len__(self):
        return self.divide_num * 3
    
    def __getitem__(self, index):
        return self.feature[index], self.label[index]

    def set_seed(self):
        torch.manual_seed(self.seed)


if __name__ == "__main__":
    dataset = E3Datasets()
    print(dataset.feature[500:1000])
    print(dataset.label[500:1000])