import torch
import torch.utils.data as torchdata
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('func', type=str)

class Data(torchdata.Dataset):
    def __init__(self, data_path):
        super(Data, self).__init__()
        self.d1, self.d2, self.d3, self.d4 = torch.load(data_path)

    def __getitem__(self, index):
        return self.d1[index], self.d2[index], self.d3[index], self.d4[index]

    def __len__(self):
        return len(self.d4)

def testdata():
    train_loader = torchdata.DataLoader(
        Data('./data/processed/train.pkl'), batch_size=128, shuffle=True, drop_last=False
    )
    dev_loader = torchdata.DataLoader(
        Data('./data/processed/dev.pkl'), batch_size=128, shuffle=True, drop_last=False
    )
    test_loader = torchdata.DataLoader(
        Data('./data/processed/test.pkl'), batch_size=128, shuffle=True, drop_last=False
    )
    for loader in [train_loader, dev_loader, test_loader]:
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


def main():
    A = parser.parse_args()
    if A.func == 'test':
        testdata()
    else:
        raise Exception('wrong args')

if __name__ == '__main__':
    main()