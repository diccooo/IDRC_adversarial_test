import torch

from builder import ModelBuilder
from config import Config

def main():
    havecuda = torch.cuda.is_available()
    torch.manual_seed(Config.seed)
    if havecuda:
        torch.cuda.manual_seed(Config.seed)

    model = ModelBuilder(havecuda)
    model.train()

if __name__ == '__main__':
    main()