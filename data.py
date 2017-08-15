import torch
import torch.utils.data as torchdata

from config import Config

class Dataset(torchdata.Dataset):
    def __init__(self, data_path, task):
        super(Dataset, self).__init__()
        self.task = task
        if self.task == 'train':
            self.d1, self.d2, self.d3, self.d4 = torch.load(data_path)
        elif self.task == 'dev' or self.task == 'test':
            self.d1, self.d2, self.d3, self.d4, self.d5 = torch.load(data_path)

    def __getitem__(self, index):
        if self.task == 'train':
            return self.d1[index], self.d2[index], self.d3[index], self.d4[index]
        elif self.task == 'dev' or self.task == 'test':
            return self.d1[index], self.d2[index], self.d3[index], self.d4[index], self.d5[index]

    def __len__(self):
        return len(self.d4)

class Data(object):
    def __init__(self, use_cuda):
        kwargs = {'batch_size':Config.batch_size, 'shuffle':Config.shuffle, 'drop_last':False}
        if use_cuda:
            kwargs['num_workers'] = 1
            kwargs['pin_memory'] = True
        train_data = Dataset('./data/processed/train.pkl', 'train')
        dev_data = Dataset('./data/processed/dev.pkl', 'dev')
        test_data = Dataset('./data/processed/test.pkl', 'test')
        self.train_size = len(train_data)
        self.dev_size = len(dev_data)
        self.test_size = len(test_data)
        # a1, a2_i, a2_a, sense
        self.train_loader = torchdata.DataLoader(train_data, **kwargs)
        # a1, a2_i, a2_a, sense1, sense2
        self.dev_loader = torchdata.DataLoader(dev_data, **kwargs)
        self.test_loader = torchdata.DataLoader(test_data, **kwargs)

def test():
    data = Data(False)
    print(data.train_size)
    print(data.dev_size)
    print(data.test_size)
    for loader in [data.train_loader, data.dev_loader, data.test_loader]:
        res = {}
        for d in loader:
            l = []
            for i in d:
                l.append(i.size())
            if tuple(l) not in res:
                res[tuple(l)] = 1
            else:
                res[tuple(l)] += 1
        for i in res:
            print(i, '*', res[i])
        print('-' * 100)

if __name__ == '__main__':
    test()