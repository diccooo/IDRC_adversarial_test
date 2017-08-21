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
        self.criterion_c = torch.nn.CrossEntropyLoss()
        self.criterion_d = torch.nn.BCELoss()
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
        total_loss = 0
        correct_n = 0
        total_loss_a = 0
        correct_n_a = 0
        for a1, a2i, a2a, sense in self.data.train_loader:
            if self.cuda:
                a1, a2i, a2a, sense = a1.cuda(), a2i.cuda(), a2a.cuda(), sense.cuda()
            a1, a2i, a2a, sense = Variable(a1), Variable(a2i), Variable(a2a), Variable(sense)

            # train i
            self.classifier.train()
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
            self.classifier.eval()
            output = self.classifier(self.a_encoder(a1, a2a))
            _, output_sense = torch.max(output, 1)
            assert output_sense.size() == sense.size()
            tmp = (output_sense == sense).long()
            correct_n_a += torch.sum(tmp).data

            loss = self.criterion_c(output, sense)
            self.a_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.a_encoder.parameters(), Config.grad_clip)
            self.a_optimizer.step()

            total_loss_a += loss.data * sense.size(0)
        return total_loss[0] / self.data.train_size, correct_n[0] / self.data.train_size, total_loss_a[0] / self.data.train_size, correct_n_a[0] / self.data.train_size

    def _adtrain_one(self):
        total_loss = 0
        correct_n = 0
        for a1, a2i, a2a, sense in self.data.train_loader:
            if self.cuda:
                a1, a2i, a2a, sense = a1.cuda(), a2i.cuda(), a2a.cuda(), sense.cuda()
            a1, a2i, a2a, sense = Variable(a1), Variable(a2i), Variable(a2a), Variable(sense)

            # phase 1, train discriminator
            self.a_encoder.eval()
            self.i_encoder.eval()
            self.discriminator.train()
            self.d_optimizer.zero_grad()
            output_i = self.discriminator(self.i_encoder(a1, a2i))
            zero_tensor = torch.zeros(output_i.size())
            if self.cuda:
                zero_tensor = zero_tensor.cuda()
            zero_tensor = Variable(zero_tensor)
            d_loss_i = self.criterion_d(output_i, zero_tensor)
            d_loss_i.backward()
            output_a = self.discriminator(self.a_encoder(a1, a2a))
            one_tensor = torch.ones(output_a.size())
            if self.cuda:
                one_tensor = one_tensor.cuda()
            one_tensor = Variable(one_tensor)
            d_loss_a = self.criterion_d(output_a, one_tensor)
            d_loss_a.backward()
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
            one_tensor = torch.ones(output_d.size())
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
        return total_loss[0] / self.data.train_size, correct_n[0] / self.data.train_size

    def _pretrain_i(self):
        best_dev_acc = 0
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
            if dev_acc >= best_dev_acc:
                best_dev_acc = dev_acc
                self._save_model(self.i_encoder, 'pre_i.pkl')
                self._save_model(self.classifier, 'pre_c.pkl')
                print('pre_i saved at epoch {}'.format(epoch))

            test_loss, test_acc = self._eval('test', 'i')
            self._print_eval('test', test_loss, test_acc)
            self.logwriter.add_scalar('loss/test_loss_i', test_loss, epoch)
            self.logwriter.add_scalar('acc/test_acc_i', test_acc*100, epoch)

    def _train_together(self):
        best_dev_acc = 0
        loss = acc = loss_a = acc_a = 0
        for epoch in range(Config.together_epochs):
            start_time = time.time()
            if epoch < Config.first_stage_epochs:
                loss, acc, loss_a, acc_a = self._pretrain_i_a_one()
            else:
                loss, acc = self._adtrain_one()
            self.logwriter.add_scalar('loss/train_loss_t', loss, epoch)
            self.logwriter.add_scalar('acc/train_acc_t', acc*100, epoch)
            self.logwriter.add_scalar('loss/train_loss_t_a', loss_a, epoch)
            self.logwriter.add_scalar('acc/train_acc_t_a', acc_a*100, epoch)

            dev_loss, dev_acc = self._eval('dev', 'i')
            dev_loss_a, dev_acc_a = self._eval('dev', 'a')
            self.logwriter.add_scalar('loss/dev_loss_t', dev_loss, epoch)
            self.logwriter.add_scalar('acc/dev_acc_t', dev_acc*100, epoch)
            self.logwriter.add_scalar('loss/dev_loss_t_a', dev_loss_a, epoch)
            self.logwriter.add_scalar('acc/dev_acc_t_a', dev_acc_a*100, epoch)
            if dev_acc >= best_dev_acc:
                best_dev_acc = dev_acc
                self._save_model(self.i_encoder, 't_i.pkl')
                self._save_model(self.classifier, 't_c.pkl')
                print('t_i t_c saved at epoch {}'.format(epoch))

            test_loss, test_acc = self._eval('test', 'i')
            test_loss_a, test_acc_a = self._eval('test', 'a')
            self.logwriter.add_scalar('loss/test_loss_t', test_loss, epoch)
            self.logwriter.add_scalar('acc/test_acc_t', test_acc*100, epoch)
            self.logwriter.add_scalar('loss/test_loss_t_a', test_loss_a, epoch)
            self.logwriter.add_scalar('acc/test_acc_t_a', test_acc_a*100, epoch)

    def train(self, need_pre_i, need_pre_a, need_final, together):
        print('start training')
        self.logwriter = SummaryWriter(Config.logdir)
        if need_pre_i:
            self._pretrain_i()
        if need_pre_a:
            if not need_pre_i:
                self._load_model(self.i_encoder, 'pre_i.pkl')
                self._load_model(self.classifier, 'pre_c.pkl')
            self._pretrain_a()
        if need_final:
            if not need_pre_a:
                if need_pre_i:
                    raise Exception('i/a not match')
                self._load_model(self.i_encoder, 'pre_i.pkl')
                self._load_model(self.a_encoder, 'pre_a.pkl')
                self._load_model(self.classifier, 'pre_c.pkl')
            self._adtrain()
        if together:
            self._train_together()
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

    def eval(self, stage):
        if stage == 'i':
            self._load_model(self.i_encoder, 'pre_i.pkl')
            self._load_model(self.classifier, 'pre_c.pkl')
            test_loss, test_acc = self._eval('test', 'i')
            self._print_eval('test', test_loss, test_acc)
        if stage == 'f':
            self._load_model(self.i_encoder, 'final_i.pkl')
            self._load_model(self.classifier, 'final_c.pkl')
            test_loss, test_acc = self._eval('test', 'i')
            self._print_eval('test', test_loss, test_acc)
        if stage == 't':
            self._load_model(self.i_encoder, 't_i.pkl')
            self._load_model(self.classifier, 't_c.pkl')
            test_loss, test_acc = self._eval('test', 'i')
            self._print_eval('test', test_loss, test_acc)
