import torch
import argparse

from ad_builder import ModelBuilder
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument('func', type=str)
parser.add_argument('stage', type=str)

def main():
    havecuda = torch.cuda.is_available()
    torch.manual_seed(Config.seed)
    if havecuda:
        torch.cuda.manual_seed(Config.seed)

    A = parser.parse_args()
    model = ModelBuilder(havecuda)
    if A.func == 'train':
        if A.stage == 'i':
            model.train('i')
        elif A.stage == 't':
            model.train('t')
        else:
            raise Exception('wrone stage')
    elif A.func == 'test':
        if A.stage == 'i':
            model.eval('i')
        elif A.stage == 't':
            model.eval('t')
        else:
            raise Exception('wrong stage')
    else:
        raise Exception('wrong func')

if __name__ == '__main__':
    main()