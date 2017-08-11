import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from config import Config

class CNNencoder(nn.Module):
    def __init__(self, we_tensor, need_kmaxavg=False):
        super(CNNencoder, self).__init__()
        self.need_kmaxavg = need_kmaxavg
        self.embed = nn.Embedding(we_tensor.size(0), we_tensor.size(1))
        self.convs = []
        for i in range(Config.conv_filter_set_num):
            self.convs.append(nn.Conv1d(
                we_tensor.size(1), Config.sent_repr_dim, Config.conv_kernel_size[i],
                stride=1, padding=Config.conv_kernel_size[i]//2
            ))
        self.nonlinear = Config.nonlinear
        self._init_weight(we_tensor)

    def _init_weight(self, we_tensor):
        self.embed.weight.data.copy_(we_tensor)
        self.embed.weight.requires_grad = False
        for i in range(Config.conv_filter_set_num):
            nn.init.normal(self.convs[i].weight, mean=0, std=0.01)
            self.convs[i].bias.data.fill_(0)

    def forward(self, input):
        embedding = self.embed(input).transpose(1, 2)
        conv_out = []
        for i in range(Config.conv_filter_set_num):
            tmp = self.nonlinear(self.convs[i](embedding))
            if tmp.size(2) > embedding.size(2):
                tmp = tmp[:, :, 1:]             # for 'same' padding
                if self.need_kmaxavg:
                    tmp = torch.topk(tmp, k=2, sorted=False)[0]
                    tmp = torch.mean(tmp, dim=2, keepdim=True)
                else:
                    tmp = torch.topk(tmp, k=1)[0]
            conv_out.append(torch.squeeze(tmp, dim=2))
            output = torch.cat(conv_out, dim=1)
        return output

class Args_encoder(nn.Module):
    def __init__(self, we_tensor, need_kmaxavg=False):
        super(Args_encoder, self).__init__()
        self.arg1enc = CNNencoder(we_tensor)
        self.arg2enc = CNNencoder(we_tensor, need_kmaxavg)
        self.dropout = nn.Dropout(Config.arg_encoder_dropout)
        self.fc = []
        if Config.arg_encoder_fc_num > 0:
            self.fc.append(nn.Linear(Config.arg_rep_dim, Config.arg_encoder_fc_dim))
            for i in range(Config.arg_encoder_fc_num - 1):
                self.fc.append(nn.Linear(Config.arg_encoder_fc_dim, Config.arg_encoder_fc_dim))
        self.nonlinear = Config.nonlinear
        self._init_weight()
    
    def _init_weight(self):
        for i in range(Config.arg_encoder_fc_num):
            self.fc[i].bias.data.fill_(0)
            nn.init.normal(self.fc[i].weight, mean=0, std=0.01)

    def forward(self, arg1, arg2):
        repr = torch.cat((self.arg1enc(arg1), self.arg2enc(arg2)), dim=1)
        for i in range(Config.arg_encoder_fc_num):
            repr = self.nonlinear(self.fc[i](self.dropout(repr)))
        return repr

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(Config.clf_dropout)
        self.fc = []
        if Config.clf_fc_num > 0:
            self.fc.append(nn.Linear(Config.pair_rep_dim, Config.clf_fc_dim))
            for i in range(Config.clf_fc_num - 1):
                self.fc.append(nn.Linear(Config.clf_fc_dim, Config.clf_fc_dim))
            lastfcdim = Config.clf_fc_dim
        else:
            lastfcdim = Config.pair_rep_dim
        self.lastfc = nn.Linear(lastfcdim, Config.clf_class_num)
        self.nonlinear = Config.nonlinear
        self._init_weight()

    def _init_weight(self):
        for i in range(Config.clf_fc_num):
            self.fc[i].bias.data.fill_(0)
            nn.init.normal(self.fc[i].weight, mean=0, std=0.01)
        self.lastfc.bias.data.fill_(0)
        nn.init.normal(self.lastfc.weight, mean=0, std=0.01)

    def forward(self, input):
        output = input
        for i in range(Config.clf_fc_num):
            output = self.nonlinear(self.fc[i](self.dropout(output)))
        output = self.lastfc(self.dropout(output))
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dropout = nn.Dropout(Config.discr_dropout)
        self.fc = []
        self.fc.append(nn.Linear(Config.pair_rep_dim, Config.discr_fc_dim))
        for i in range(3):
            self.fc.append(nn.Linear(Config.discr_fc_dim, Config.discr_fc_dim))
        self.gatefc = []
        for i in range(2):
            self.gatefc.append(nn.Linear(Config.discr_fc_dim, Config.discr_fc_dim))
        self.lastfc = nn.Linear(Config.discr_fc_dim, 2)
        self.nonlinear = Config.nonlinear
        self._init_weight()

    def _init_weight(self):
        for i in range(4):
            self.fc[i].bias.data.fill_(0)
            nn.init.normal(self.fc[i].weight, mean=0, std=0.01)
        for i in range(2):
            self.gatefc[i].bias.data.fill_(0)
            nn.init.normal(self.gatefc[i].weight, mean=0, std=0.01)
        self.lastfc.bias.data.fill_(0)
        nn.init.normal(self.lastfc.weight, mean=0, std=0.01)

    def forward(self, input):
        o1 = self.nonlinear(self.fc[0](self.dropout(input)))
        o2 = self.nonlinear(self.fc[1](self.dropout(o1)))
        g0 = F.sigmoid(self.gatefc[0](self.dropout(o1)))
        g1 = F.sigmoid(self.gatefc[1](self.dropout(o1)))
        o3 = self.nonlinear(self.fc[2](self.dropout(torch.mul(o2, g1))))
        o4 = self.nonlinear(self.fc[3](self.dropout(torch.mul(o3, g0))))
        output = self.lastfc(self.dropout(o4))
        return output


def testCNN(need_kmaxavg=False, need_print=False):
    we = torch.load('./datas/data/we.pkl')
    model = CNNencoder(we, need_kmaxavg)
    model.eval()
    input = np.random.randint(0, we.size(0), (5, 80))
    input = torch.from_numpy(input)
    input = Variable(input)
    out = model(input)
    if need_print:
        print(input.size())
        print(out.size())
    else:
        return out

def testArgenc(need_kmaxavg=False, need_print=False):
    we = torch.load('./datas/data/we.pkl')
    model = Args_encoder(we, need_kmaxavg)
    model.eval()
    arg1 = np.random.randint(0, we.size(0), (5, 80))
    arg1 = Variable(torch.from_numpy(arg1))
    arg2 = np.random.randint(0, we.size(0), (5, 80))
    arg2 = Variable(torch.from_numpy(arg2))
    out = model(arg1, arg2)
    if need_print:
        print(out)
    else:
        return out

def testClf():
    model = Classifier()
    model.eval()
    input = testArgenc()
    print(input.size())
    out = model(input)
    print(out)

def testDiscr():
    model = Discriminator()
    model.eval()
    input = testArgenc()
    print(input.size())
    out = model(input)
    print(out)

def test():
    testDiscr()

if __name__ == '__main__':
    test()