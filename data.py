import torch
import torch.utils.data as torchdata

from config import Config

class Dataset(torchdata.Dataset):
    def __init__(self, data_path):
        super(Dataset, self).__init__()
        self.d1, self.d2, self.d3, self.d4 = torch.load(data_path)

    def __getitem__(self, index):
        return self.d1[index], self.d2[index], self.d3[index], self.d4[index]

    def __len__(self):
        return len(self.d4)

class Data(object):
    def __init__(self, use_cuda):
        kwargs = {'batch_size':Config.batch_size, 'shuffle':Config.shuffle, 'drop_last':False}
        if use_cuda:
            kwargs['num_workers'] = 1
            kwargs['pin_memory'] = True
        # a1, a2_i, a2_a, sense
        self.train_loader = torchdata.DataLoader(Dataset('./data/processed/train.pkl'), **kwargs)
        # a1, a2, sense1, sense2
        self.dev_loader = torchdata.DataLoader(Dataset('./data/processed/dev.pkl'), **kwargs)
        self.test_loader = torchdata.DataLoader(Dataset('./data/processed/test.pkl'), **kwargs)

def test():
    data = Data(False)
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