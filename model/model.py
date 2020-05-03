import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # nn.init.xavier_normal_(m.weight.data)
        m.weight.data.normal_(0.0, 1e-3)
    elif classname.find('Linear') != -1:
        # nn.init.xavier_normal_(m.weight.data)
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0)


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            # nn.Linear(input_size, 1280),
            # nn.PReLU(),
            # nn.Dropout(),
            # nn.Linear(1280, 1024),
            # nn.PReLU(),
            # nn.Dropout(),
            # nn.Linear(1024, 896),
            # nn.PReLU(),
            # nn.Dropout(),
            # nn.Linear(896, 768),
            # nn.PReLU(),
            # nn.Dropout(),
            # nn.Linear(768, 512),
            # nn.PReLU(),
            # nn.Dropout(),
            # nn.Linear(512, 384),
            # nn.PReLU(),
            # nn.Dropout(),
            # nn.Linear(384, 256),
            # nn.PReLU(),
            # nn.Dropout(),
            # nn.Linear(256, 256),
            # nn.PReLU(),
            # nn.Dropout(),
            nn.Linear(input_size, 256),
            nn.PReLU(),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, output_size),
            nn.Tanh(),
        )
        # self.fc.apply(weights_init)

    def forward(self, x):
        out = self.fc(x)
        return out
