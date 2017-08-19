import torch
import argparse

from builder import ModelBuilder
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
            model.train(True, False, False, False)
        elif A.stage == 'a':
            model.train(False, True, False, False)
        elif A.stage == 'f':
            model.train(False, False, True, False)
        elif A.stage == 'ia':
            model.train(True, True, False, False)
        elif A.stage == 'af':
            model.train(False, True, True, False)
        elif A.stage == 'iaf':
            model.train(True, True, True, False)
        elif A.stage == 't':
            model.train(False, False, False, True)
        else:
            raise Exception('wrone stage')
    elif A.func == 'test':
        if A.stage == 'i':
            model.eval('i')
        elif A.stage == 'f':
            model.eval('f')
        elif A.stage == 't':
            model.eval('t')
        else:
            raise Exception('wrong stage')
    else:
        raise Exception('wrong func')

if __name__ == '__main__':
    main()