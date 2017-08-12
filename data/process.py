import sys
sys.path.append('../')
import numpy as np
import torch
import gensim
import pickle
from tqdm import tqdm
import argparse

from config import Config

parser = argparse.ArgumentParser()
parser.add_argument('func', type=str)

class PreData(object):
    def __init__(self):
        self._read_text()
        self._get_vocab()
        self._get_we()
        self._pre_data()
        print('predata done.')

    def _read_text(self):
        print('reading text...')
        if Config.corpus_splitting == 1:
            path_pre = './interim/lin/'
        elif Config.corpus_splitting == 2:
            path_pre = './interim/ji/'
        with open(path_pre + 'train.pkl', 'rb') as f:
            self.arg1_train_r = pickle.load(f)
            self.arg2_train_r = pickle.load(f)
            self.conn_train_r = pickle.load(f)
            self.sense_train_r = pickle.load(f)
        with open(path_pre + 'dev.pkl', 'rb') as f:
            self.arg1_dev_r = pickle.load(f)
            self.arg2_dev_r = pickle.load(f)
            self.sense1_dev_r = pickle.load(f)
            self.sense2_dev_r = pickle.load(f)
        with open(path_pre + 'test.pkl', 'rb') as f:
            self.arg1_test_r = pickle.load(f)
            self.arg2_test_r = pickle.load(f)
            self.sense1_test_r = pickle.load(f)
            self.sense2_test_r = pickle.load(f)

    def _get_vocab(self):
        print('geting vocab...')
        self.word2i = {'<unk>':0, '</s>':1}
        self.v_size = 2
        for sentlist in [
            self.arg1_train_r, self.arg2_train_r, self.conn_train_r,
            self.arg1_dev_r, self.arg2_dev_r, self.arg1_test_r, self.arg2_test_r
        ]:
            for sent in sentlist:
                for word in sent:
                    if word not in self.word2i:
                        self.word2i[word] = self.v_size
                        self.v_size += 1

    def _get_we(self):
        print('reading pretrained w2v...')
        w2v = gensim.models.KeyedVectors.load_word2vec_format(Config.wordvec_path, binary=True)
        pretrained_vocab = w2v.vocab.keys()
        print('making we...')
        we = np.zeros((self.v_size, Config.wordvec_dim))
        for word, idx in self.word2i.items():
            if word in pretrained_vocab:
                we[idx, :] = w2v[word]
        self.we = torch.from_numpy(we)
        torch.save(self.we, './processed/we.pkl')

    def _text2i(self, texts):
        l = len(texts)
        tensor = torch.LongTensor(l, Config.max_sent_len).zero_()
        for i in tqdm(range(l)):
                s = texts[i] + ['</s>']
                minlen = min(len(s), Config.max_sent_len)
                for j in range(minlen):
                    tensor[i][j] = self.word2i[s[j]]
        return tensor

    def _sense2i(self, senses):
        l = len(senses)
        tensor = torch.LongTensor(l)
        for i in tqdm(range(l)):
            if senses[i][0] is None:
                tensor[i] = -1
            else:
                tensor[i] = Config.sense2i[senses[i][0]]
        return tensor

    def _pre_data(self):
        print('pre training data...')
        # a1, a2_i, a2_a, sense
        train_data = [
            self._text2i(self.arg1_train_r),
            self._text2i(self.arg2_train_r),
            self._text2i([i+j for (i, j) in zip(self.conn_train_r, self.arg2_train_r)]),
            self._sense2i(self.sense_train_r)
        ]
        print('pre dev/test data...')
        # a1, a2, sense1, sense2
        dev_data = [
            self._text2i(self.arg1_dev_r),
            self._text2i(self.arg2_dev_r),
            self._sense2i(self.sense1_dev_r),
            self._sense2i(self.sense2_dev_r)
        ]
        test_data = [
            self._text2i(self.arg1_test_r),
            self._text2i(self.arg2_test_r),
            self._sense2i(self.sense1_test_r),
            self._sense2i(self.sense2_test_r)
        ]
        print('saving data...')
        torch.save(train_data, './processed/train.pkl')
        torch.save(dev_data, './processed/dev.pkl')
        torch.save(test_data, './processed/test.pkl')

def testpredata():
    we = torch.load('./processed/we.pkl')
    train_data = torch.load('./processed/train.pkl')
    dev_data = torch.load('./processed/dev.pkl')
    test_data = torch.load('./processed/test.pkl')
    print(we)
    for data in [train_data, dev_data, test_data]:
        for d in data:
            print(d.size())

def main():
    A = parser.parse_args()
    if A.func == 'pre':
        d = PreData()
    elif A.func == 'test':
        testpredata()
    else:
        raise Exception('wrong args')

if __name__ == '__main__':
    main()