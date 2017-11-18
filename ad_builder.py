import time
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from data import Data
from model import CNN_Args_encoder, Classifier, Discriminator
from config import Config

class ModelBuilder(object):
    def __init__(self, use_cuda):
        self.cuda = use_cuda
        self._pre_data()
        self._build_model()
        self.i_mb = 0
    
    def _pre_data(self):
        print('pre data...')
        self.data = Data(self.cuda)

    def _build_model(self):
        print('building model...')
        we = torch.load('./data/processed/we.pkl')
        self.i_encoder = CNN_Args_encoder(we)
        self.a_encoder = CNN_Args_encoder(we, need_kmaxavg=True)
        self.classifier = Classifier()
        self.discriminator = Discriminator()
        if self.cuda:
            self.i_encoder.cuda()
            self.a_encoder.cuda()
            self.classifier.cuda()
            self.discriminator.cuda()
        self.criterion_c = torch.nn.CrossEntropyLoss()
        self.criterion_d = torch.nn.BCELoss()
        para_filter = lambda model: filter(lambda p: p.requires_grad, model.parameters())
        self.i_optimizer = torch.optim.Adagrad(para_filter(self.i_encoder), Config.lr, weight_decay=Config.l2_penalty)
        self.a_optimizer = torch.optim.Adagrad(para_filter(self.a_encoder), Config.lr, weight_decay=Config.l2_penalty)
        self.c_optimizer = torch.optim.Adagrad(self.classifier.parameters(), Config.lr, weight_decay=Config.l2_penalty)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), Config.lr_d, weight_decay=Config.l2_penalty)

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
            
            loss = self.criterion_c(output, sense)
            self.i_optimizer.zero_grad()
            self.c_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.i_encoder.parameters(), Config.grad_clip)
            torch.nn.utils.clip_grad_norm(self.classifier.parameters(), Config.grad_clip)
            self.i_optimizer.step()
            self.c_optimizer.step()
            
            total_loss += loss.data * sense.size(0)
        return total_loss[0] / self.data.train_size, correct_n[0] / self.data.train_size

    def _pretrain_i_a_one(self):
        self.i_encoder.train()
        self.a_encoder.train()
        self.classifier.train()
        total_loss = 0
        correct_n = 0
        total_loss_a = 0
        correct_n_a = 0
        for a1, a2i, a2a, sense in self.data.train_loader:
            if self.cuda:
                a1, a2i, a2a, sense = a1.cuda(), a2i.cuda(), a2a.cuda(), sense.cuda()
            a1, a2i, a2a, sense = Variable(a1), Variable(a2i), Variable(a2a), Variable(sense)

            # train i
            output = self.classifier(self.i_encoder(a1, a2i))
            _, output_sense = torch.max(output, 1)
            assert output_sense.size() == sense.size()
            tmp = (output_sense == sense).long()
            correct_n += torch.sum(tmp).data
            
            loss = self.criterion_c(output, sense)
            self.i_optimizer.zero_grad()
            self.c_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.i_encoder.parameters(), Config.grad_clip)
            torch.nn.utils.clip_grad_norm(self.classifier.parameters(), Config.grad_clip)
            self.i_optimizer.step()
            self.c_optimizer.step()
            
            total_loss += loss.data * sense.size(0)

            #train a
            output = self.classifier(self.a_encoder(a1, a2a))
            _, output_sense = torch.max(output, 1)
            assert output_sense.size() == sense.size()
            tmp = (output_sense == sense).long()
            correct_n_a += torch.sum(tmp).data

            loss = self.criterion_c(output, sense)
            self.a_optimizer.zero_grad()
            self.c_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.a_encoder.parameters(), Config.grad_clip)
            torch.nn.utils.clip_grad_norm(self.classifier.parameters(), Config.grad_clip)
            self.a_optimizer.step()
            self.c_optimizer.step()

            total_loss_a += loss.data * sense.size(0)
        return total_loss[0] / self.data.train_size, correct_n[0] / self.data.train_size, total_loss_a[0] / self.data.train_size, correct_n_a[0] / self.data.train_size

    def _adtrain_one(self, acc_d_for_train):
        total_loss = 0
        total_loss_2 = 0
        correct_n = 0
        correct_n_d = 0
        correct_n_d_for_train = 0
        for a1, a2i, a2a, sense in self.data.train_loader:
            if self.cuda:
                a1, a2i, a2a, sense = a1.cuda(), a2i.cuda(), a2a.cuda(), sense.cuda()
            a1, a2i, a2a, sense = Variable(a1), Variable(a2i), Variable(a2a), Variable(sense)

            # phase 1, train discriminator
            flag = 0
            for k in range(Config.kd):
                # if self._test_d() != 1:
                if True:
                    temp_d = 0
                    self.a_encoder.eval()
                    self.i_encoder.eval()
                    self.discriminator.train()
                    self.d_optimizer.zero_grad()
                    output_i = self.discriminator(self.i_encoder(a1, a2i))
                    temp_d += torch.sum((output_i < 0.5).long()).data
                    # zero_tensor = torch.zeros(output_i.size())
                    zero_tensor = torch.Tensor(output_i.size()).random_(0,100) * 0.003
                    if self.cuda:
                        zero_tensor = zero_tensor.cuda()
                    zero_tensor = Variable(zero_tensor)
                    d_loss_i = self.criterion_d(output_i, zero_tensor)
                    d_loss_i.backward()
                    output_a = self.discriminator(self.a_encoder(a1, a2a))
                    temp_d += torch.sum((output_a >= 0.5).long()).data
                    # one_tensor = torch.ones(output_a.size())
                    # one_tensor = torch.Tensor(output_a.size()).fill_(Config.alpha)
                    one_tensor = torch.Tensor(output_a.size()).random_(0,100) * 0.005 + 0.7
                    if self.cuda:
                        one_tensor = one_tensor.cuda()
                    one_tensor = Variable(one_tensor)
                    d_loss_a = self.criterion_d(output_a, one_tensor)
                    d_loss_a.backward()
                    correct_n_d_for_train += temp_d
                    temp_d = max(temp_d[0] / sense.size(0) / 2, acc_d_for_train)
                    if temp_d < Config.thresh_high:
                        torch.nn.utils.clip_grad_norm(self.discriminator.parameters(), Config.grad_clip)
                        self.d_optimizer.step()
        
            # phase 2, train i/c
            self.i_encoder.train()
            self.classifier.train()
            self.discriminator.eval()
            self.i_optimizer.zero_grad()
            self.c_optimizer.zero_grad()
            sent_repr = self.i_encoder(a1, a2i)

            output = self.classifier(sent_repr)
            _, output_sense = torch.max(output, 1)
            assert output_sense.size() == sense.size()
            tmp = (output_sense == sense).long()
            correct_n += torch.sum(tmp).data
            loss_1 = self.criterion_c(output, sense)

            output_d = self.discriminator(sent_repr)
            correct_n_d += torch.sum((output_d < 0.5).long()).data
            one_tensor = torch.ones(output_d.size())
            # one_tensor = torch.Tensor(output_d.size()).fill_(Config.alpha)
            # one_tensor = torch.Tensor(output_d.size()).random_(0,100) * 0.005 + 0.7
            if self.cuda:
                one_tensor = one_tensor.cuda()
            one_tensor = Variable(one_tensor)
            loss_2 = self.criterion_d(output_d, one_tensor)

            loss = loss_1 + loss_2 * Config.lambda1
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.i_encoder.parameters(), Config.grad_clip)
            torch.nn.utils.clip_grad_norm(self.classifier.parameters(), Config.grad_clip)
            self.i_optimizer.step()
            self.c_optimizer.step()
            
            total_loss += loss.data * sense.size(0)
            total_loss_2 += loss_2.data * sense.size(0)

            test_loss, test_acc = self._eval('test', 'i')
            self.logwriter.add_scalar('acc/test_acc_t_mb', test_acc*100, self.i_mb)
            self.i_mb += 1

        return total_loss[0] / self.data.train_size, correct_n[0] / self.data.train_size, correct_n_d[0] / self.data.train_size, total_loss_2[0] / self.data.train_size, correct_n_d_for_train[0] / self.data.train_size / 2

    def _pretrain_i(self):
        best_test_acc = 0
        for epoch in range(Config.pre_i_epochs):
            start_time = time.time()
            loss, acc = self._pretrain_i_one()
            self._print_train(epoch, time.time()-start_time, loss, acc)
            self.logwriter.add_scalar('loss/train_loss_i', loss, epoch)
            self.logwriter.add_scalar('acc/train_acc_i', acc*100, epoch)

            dev_loss, dev_acc = self._eval('dev', 'i')
            self._print_eval('dev', dev_loss, dev_acc)
            self.logwriter.add_scalar('loss/dev_loss_i', dev_loss, epoch)
            self.logwriter.add_scalar('acc/dev_acc_i', dev_acc*100, epoch)

            test_loss, test_acc = self._eval('test', 'i')
            self._print_eval('test', test_loss, test_acc)
            self.logwriter.add_scalar('loss/test_loss_i', test_loss, epoch)
            self.logwriter.add_scalar('acc/test_acc_i', test_acc*100, epoch)
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                self._save_model(self.i_encoder, 'i.pkl')
                self._save_model(self.classifier, 'c.pkl')
                print('i_model saved at epoch {}'.format(epoch))

    def _adjust_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _train_together(self):
        best_test_acc = 0
        loss = acc = loss_a = acc_a = 0
        lr_t = Config.lr_t
        acc_d_for_train = 0
        for epoch in range(Config.together_epochs):
            start_time = time.time()
            if epoch < Config.first_stage_epochs:
                loss, acc, loss_a, acc_a = self._pretrain_i_a_one()
            else:
                if epoch == Config.first_stage_epochs:
                    self._adjust_learning_rate(self.i_optimizer, lr_t)
                    self._adjust_learning_rate(self.c_optimizer, lr_t / 2)
                # elif (epoch - Config.first_stage_epochs) % 20 == 0:
                #     lr_t *= 0.8
                #     self._adjust_learning_rate(self.i_optimizer, lr_t)
                #     self._adjust_learning_rate(self.c_optimizer, lr_t)
                loss, acc, acc_d, loss_2, acc_d_for_train = self._adtrain_one(acc_d_for_train)
            self._print_train(epoch, time.time()-start_time, loss, acc)
            self.logwriter.add_scalar('loss/train_loss_t', loss, epoch)
            self.logwriter.add_scalar('acc/train_acc_t', acc*100, epoch)
            self.logwriter.add_scalar('loss/train_loss_t_a', loss_a, epoch)
            self.logwriter.add_scalar('acc/train_acc_t_a', acc_a*100, epoch)
            if epoch >= Config.first_stage_epochs:
                self.logwriter.add_scalar('acc/train_acc_d', acc_d*100, epoch)
                self.logwriter.add_scalar('loss/train_loss_2', loss_2, epoch)
                self.logwriter.add_scalar('acc/acc_d_for_train', acc_d_for_train*100, epoch)

            dev_loss, dev_acc = self._eval('dev', 'i')
            dev_loss_a, dev_acc_a = self._eval('dev', 'a')
            self._print_eval('dev', dev_loss, dev_acc)
            self.logwriter.add_scalar('loss/dev_loss_t', dev_loss, epoch)
            self.logwriter.add_scalar('acc/dev_acc_t', dev_acc*100, epoch)
            self.logwriter.add_scalar('loss/dev_loss_t_a', dev_loss_a, epoch)
            self.logwriter.add_scalar('acc/dev_acc_t_a', dev_acc_a*100, epoch)
            if epoch >= Config.first_stage_epochs:
                dev_acc_d = self._eval_d('dev')
                self.logwriter.add_scalar('acc/dev_acc_d', dev_acc_d*100, epoch)

            test_loss, test_acc = self._eval('test', 'i')
            test_loss_a, test_acc_a = self._eval('test', 'a')
            self._print_eval('test', test_loss, test_acc)
            self.logwriter.add_scalar('loss/test_loss_t', test_loss, epoch)
            self.logwriter.add_scalar('acc/test_acc_t', test_acc*100, epoch)
            self.logwriter.add_scalar('loss/test_loss_t_a', test_loss_a, epoch)
            self.logwriter.add_scalar('acc/test_acc_t_a', test_acc_a*100, epoch)
            if epoch >= Config.first_stage_epochs:
                test_acc_d = self._eval_d('test')
                self.logwriter.add_scalar('acc/test_acc_d', test_acc_d*100, epoch)
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                self._save_model(self.i_encoder, 't_i.pkl')
                self._save_model(self.classifier, 't_c.pkl')
                print('t_i t_c saved at epoch {}'.format(epoch))

    def train(self, i_or_t):
        print('start training')
        self.logwriter = SummaryWriter(Config.logdir)
        if i_or_t == 'i':
            self._pretrain_i()
        elif i_or_t == 't':
            self._train_together()
        else:
            raise Exception('wrong i_or_t')
        print('training done')

    def _eval(self, task, i_or_a):
        self.i_encoder.eval()
        self.a_encoder.eval()
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

            loss = self.criterion_c(output, gold_sense)
            total_loss += loss.data * gold_sense.size(0)
        return total_loss[0] / n, correct_n[0] / n

    def _eval_d(self, task):
        self.i_encoder.eval()
        self.a_encoder.eval()
        self.classifier.eval()
        correct_n = 0
        if task == 'train':
            n = self.data.train_size
            for a1, a2i, a2a, sense in self.data.train_loader:
                if self.cuda:
                    a1, a2i, a2a, sense = a1.cuda(), a2i.cuda(), a2a.cuda(), sense.cuda()
                a1 = Variable(a1, volatile=True)
                a2i = Variable(a2i, volatile=True)
                a2a = Variable(a2a, volatile=True)
                sense = Variable(sense, volatile=True)
                
                output_i = self.discriminator(self.i_encoder(a1, a2i))
                correct_n += torch.sum((output_i < 0.5).long()).data
                # output_a = self.discriminator(self.a_encoder(a1, a2a))
                # correct_n += torch.sum((output_a >= 0.5).long()).data
        else:
            if task == 'dev':
                data = self.data.dev_loader
                n = self.data.dev_size
            elif task == 'test':
                data = self.data.test_loader
                n = self.data.test_size
            for a1, a2i, a2a, sense1, sense2 in data:
                if self.cuda:
                    a1, a2i, a2a, sense1, sense2 = a1.cuda(), a2i.cuda(), a2a.cuda(), sense1.cuda(), sense2.cuda()
                a1 = Variable(a1, volatile=True)
                a2i = Variable(a2i, volatile=True)
                a2a = Variable(a2a, volatile=True)
                sense1 = Variable(sense1, volatile=True)
                sense2 = Variable(sense2, volatile=True)
                
                output_i = self.discriminator(self.i_encoder(a1, a2i))
                correct_n += torch.sum((output_i < 0.5).long()).data
                # output_a = self.discriminator(self.a_encoder(a1, a2a))
                # correct_n += torch.sum((output_a >= 0.5).long()).data
        return correct_n[0] / n

    def _test_d(self):
        acc = self._eval_d('dev')
        phase = -100
        if acc >= Config.thresh_high:
            phase = 1
        elif acc > Config.thresh_low:
            phase = 0
        else:
            phase = -1
        return phase

    def eval(self, stage):
        if stage == 'i':
            self._load_model(self.i_encoder, 'i.pkl')
            self._load_model(self.classifier, 'c.pkl')
            test_loss, test_acc = self._eval('test', 'i')
            self._print_eval('test', test_loss, test_acc)
        elif stage == 't':
            self._load_model(self.i_encoder, 't_i.pkl')
            self._load_model(self.classifier, 't_c.pkl')
            test_loss, test_acc = self._eval('test', 'i')
            self._print_eval('test', test_loss, test_acc)
        else:
            raise Exception('wrong eval stage')
