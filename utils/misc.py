import torch
from torch.autograd import Variable
import copy

import time

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def save_state(net, torch_seed, np_seed, py_seed, fname):
    # save both model state and optimizer state
    states = {
        'state_dict': net.state_dict(),
        'optimizer': net.opt.state_dict(),
        'torch_seed': torch_seed,
        'np_seed': np_seed,
        'py_seed': py_seed
    }
    torch.save(states, fname)


def load_net_state(net, fname):
    checkpoint = torch.load(fname,
                            map_location='cuda:%d' %
                            (torch.cuda.current_device()))
    net.load_state_dict(checkpoint['state_dict'])


def load_opt_state(net, fname):
    checkpoint = torch.load(fname,
                            map_location='cuda:%d' %
                            (torch.cuda.current_device()))
    net.opt.load_state_dict(checkpoint['optimizer'])


def load_seed(fname):
    # load both torch random seed, and numpy random seed
    checkpoint = torch.load(fname,
                            map_location='cuda:%d' %
                            (torch.cuda.current_device()))
    return checkpoint['torch_seed'], checkpoint['np_seed'], checkpoint[
        'py_seed']


def normalize(x, bound, time_flag=False):
    # normalize to -1 ~ 1  (bound can be a tensor)
    #return x
    bound = torch.tensor(bound)
    if len(x[0]) != len(bound):
        x[..., :2] = (x[..., :2]/bound[:2])*2 - 1
        x[..., 2] = x[..., 2]/bound[2]
        x[..., 3:5] = (x[..., 3:5]/bound[:2])*2 - 1
        x[..., 5] = x[..., 5]/bound[2]
    else:
        x = (x / bound)
        x[:, :2] = x[:,:2]*2-1
    return x


def unnormalize(x, bound, time_flag=False):
    # normalize to -1 ~ 1  (bound can be a tensor)
    # x only one dim
    #return x
    time_0 = time.time()
    bound = torch.tensor(bound)
    if len(x) != len(bound):
        # then the proceding is obstacle
        # don't normalize obstacles
        x[:, :-2 * len(bound)] = (x[:, :-2 * len(bound)]+1) * bound[0]
        x[:, -2 *
          len(bound):-len(bound)] = x[:, -2 * len(bound):-len(bound)] * bound
        x[:, -len(bound):] = x[:, -len(bound):] * bound
    else:
        x = (x+1) * bound
    if time_flag:
        return x, time.time() - time_0
    else:
        return x
