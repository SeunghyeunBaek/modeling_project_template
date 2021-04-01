import torch.optim as optim

def get_optimizer(optimizer_str: str)-> 'optimizer':

    if optimizer_str == 'sgd':

        optimizer = optim.SGD

    return optimizer


if __name__ == '__main__':
    pass
