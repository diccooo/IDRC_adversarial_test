import time
import torch
from torch.autograd import Variable
from tensorboard import SummaryWriter

from data import Data
from model import Args_encoder, Classifier, Discriminator
from config import Config

class ModelBuilder(object):
    def __init__(self, use_cuda):
        self.cuda = use_cuda
        self._pre_data()
        self._build_model()
        self.logwriter = SummaryWriter(Config.logdir)
    
    def _pre_data(self):
        print('pre data...')
        self.data = Data(self.cuda)

    def _build_model(self):
        print('building model...')
        we = torch.load('./data/processed/we.pkl')
        self.i_encoder = Args_encoder(we)
        self.a_encoder = Args_encoder(we, need_kmaxavg=True)
        self.classifier = Classifier()
        self.discriminator = Discriminator()
        if self.cuda:
            self.i_encoder.cuda()
            self.a_encoder.cuda()
            self.classifier.cuda()
            self.discriminator.cuda()
        self.criterion = torch.nn.CrossEntropyLoss()
        para_filter = lambda model: filter(lambda p: p.requires_grad, model.parameters())
        self.i_optimizer = torch.optim.Adagrad(para_filter(self.i_encoder), Config.lr, weight_decay=Config.l2_penalty)
        self.a_optimizer = torch.optim.Adagrad(para_filter(self.a_encoder), Config.lr, weight_decay=Config.l2_penalty)
        self.c_optimizer = torch.optim.Adagrad(self.classifier.parameters(), Config.lr, weight_decay=Config.l2_penalty)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), Config.lr, weight_decay=Config.l2_penalty)

    def _print_train(self, epoch, time, loss, acc):
        print('-' * 80)
        print(
            '| end of epoch {:3d} | time: {:5.2f}s | loss: {:10.5f} | acc: {:5.2f}% |'.format(
                epoch, time, loss, acc * 100
            )
        )
        print('-' * 80)

    def _print_eval(self, task, loss, acc):
        print(
            '| ' + task + ' loss {:10.5f} | acc {:5.2f}% |'.format(loss, acc * 100)
        )
        print('-' * 80)

    def _save_model(self, model, filename):
        torch.save(model.state_dict(), './weights/' + filename)

    def _load_model(self, model, filename):
        model.load_state_dict(torch.load('./weights/' + filename))

    def _pretrain_i_one(self):
        self.i_encoder.train()
        self.classifier.train()
        total_loss = 0
        correct_n = 0
        for a1, a2i, a2a, sense in self.data.train_loader:
            if self.cuda:
                a1, a2i, a2a, sense = a1.cuda(), a2i.cuda(), a2a.cuda(), sense.cuda()
            a1, a2i, a2a, sense = Variable(a1), Variable(a2i), Variable(a2a), Variable(sense)
            
            output = self.classifier(self.i_encoder(a1, a2i))
            _, output_sense = torch.max(output, 1)
            assert output_sense.size() == sense.size()
            tmp = (output_sense == sense).long()
            correct_n += torch.sum(tmp).data
            
            loss = self.criterion(output, sense)
            self.i_optimizer.zero_grad()
            self.c_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.i_encoder.parameters(), Config.grad_clip)
            torch.nn.utils.clip_grad_norm(self.classifier.parameters(), Config.grad_clip)
            self.i_optimizer.step()
            self.c_optimizer.step()
            
            total_loss += loss.data * sense.size(0)
        return total_loss[0] / self.data.train_size, correct_n[0] / self.data.train_size

    def _pretrain_a_one(self):
        self.a_encoder.train()
        self.classifier.eval()
        total_loss = 0
        correct_n = 0
        for a1, a2i, a2a, sense in self.data.train_loader:
            if self.cuda:
                a1, a2i, a2a, sense = a1.cuda(), a2i.cuda(), a2a.cuda(), sense.cuda()
            a1, a2i, a2a, sense = Variable(a1), Variable(a2i), Variable(a2a), Variable(sense)
            
            output = self.classifier(self.a_encoder(a1, a2a))
            _, output_sense = torch.max(output, 1)
            assert output_sense.size() == sense.size()
            tmp = (output_sense == sense).long()
            correct_n += torch.sum(tmp).data

            loss = self.criterion(output, sense)
            self.a_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.a_encoder.parameters(), Config.grad_clip)
            self.a_optimizer.step()

            total_loss += loss.data * sense.size(0)
        return total_loss[0] / self.data.train_size, correct_n[0] / self.data.train_size

    def _adtrain_one(self):
        pass

    def _pretrain(self):
        for epoch in range(1,301):
            start_time = time.time()
            loss, acc = self._pretrain_i_one()
            self._print_train(epoch, time.time()-start_time, loss, acc)
            self.logwriter.add_scalar('loss/train_loss', loss, epoch)
            self.logwriter.add_scalar('acc/train_acc', acc*100, epoch)

            dev_loss, dev_acc = self._eval('dev', 'i')
            self._print_eval('dev', dev_loss, dev_acc)
            self.logwriter.add_scalar('loss/dev_loss', dev_loss, epoch)
            self.logwriter.add_scalar('acc/dev_acc', dev_acc*100, epoch)

            test_loss, test_acc = self._eval('test', 'i')
            self._print_eval('test', test_loss, test_acc)
            self.logwriter.add_scalar('loss/test_loss', test_loss, epoch)
            self.logwriter.add_scalar('acc/test_acc', test_acc*100, epoch)
            
        print('training done')

    def _adtrain(self):
        pass

    def train(self):
        print('start training')
        self._pretrain()

    def _eval(self, task, i_or_a):
        self.i_encoder.eval()
        self.classifier.eval()
        total_loss = 0
        correct_n = 0
        if task == 'dev':
            data = self.data.dev_loader
            n = self.data.dev_size
        elif task == 'test':
            data = self.data.test_loader
            n = self.data.test_size
        else:
            raise Exception('wrong eval task')
        for a1, a2i, a2a, sense1, sense2 in data:
            if self.cuda:
                a1, a2i, a2a, sense1, sense2 = a1.cuda(), a2i.cuda(), a2a.cuda(), sense1.cuda(), sense2.cuda()
            a1 = Variable(a1, volatile=True)
            a2i = Variable(a2i, volatile=True)
            a2a = Variable(a2a, volatile=True)
            sense1 = Variable(sense1, volatile=True)
            sense2 = Variable(sense2, volatile=True)

            if i_or_a == 'i':
                output = self.classifier(self.i_encoder(a1, a2i))
            elif i_or_a == 'a':
                output = self.classifier(self.a_encoder(a1, a2a))
            else:
                raise Exception('wrong i_or_a')
            _, output_sense = torch.max(output, 1)
            assert output_sense.size() == sense1.size()
            gold_sense = sense1
            mask = (output_sense == sense2)
            gold_sense[mask] = sense2[mask]
            tmp = (output_sense == gold_sense).long()
            correct_n += torch.sum(tmp).data

            loss = self.criterion(output, gold_sense)
            total_loss += loss.data * gold_sense.size(0)
        return total_loss[0] / n, correct_n[0] / n

    def eval(self):
        pass