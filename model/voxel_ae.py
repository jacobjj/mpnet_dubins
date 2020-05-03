import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_, clip_grad_value_


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # nn.init.xavier_normal_(m.weight.data)
        m.weight.data.normal_(0.0, 1e-3)
    elif classname.find('Linear') != -1:
        # nn.init.xavier_normal_(m.weight.data)
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0)


class Encoder(nn.Module):
    """
    A 2D VoxNet for encoding obstacle space
    """

    def __init__(self, output_size, state_size, input_size):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=8,
                      kernel_size=[5, 5],
                      stride=[1, 1]),
            # nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=3),
            nn.PReLU(),
            # nn.ReLU(),
            nn.Conv2d(in_channels=8,
                      out_channels=16,
                      kernel_size=[3, 3],
                      stride=[1, 1]),
            # nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=3),
            nn.PReLU(),
            # nn.ReLU(),
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=[3, 3],
                      stride=[1, 1]),
            # nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        # self.encoder.apply(weights_init)
        # For accepting different input shapes
        x = self.encoder(torch.autograd.Variable(torch.rand([1] + input_size)))
        first_fc_in_features = 1
        for n in x.size()[1:]:
            first_fc_in_features *= n
        self.head = nn.Sequential(
            nn.Linear(first_fc_in_features + state_size + 1, 256),
            nn.PReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Dropout(),
            nn.Linear(256, output_size),
        )
        # self.head.apply(weights_init)

    @torch.jit.ignore
    def get_contractive_loss(self):
        """
        Return contractive loss for the outer layer
        """
        keys = list(self.head.state_dict().keys())
        W = Variable(self.head.state_dict()[keys[-2]])
        if torch.cuda.is_available():
            W = W.cuda()
        contractive_loss = torch.sum(W**2, dim=1).sum()
        return contractive_loss

    
    def forward(self, obs, state):
        obs = self.encoder(obs)
        x = obs.view(obs.size(0), -1)
        relative_target = state[:, 3:5] - state[:, :2]
        input_state = torch.cat(
            (
                relative_target,
                state[:, 5].reshape(-1, 1),
                state[:, 2].reshape(-1, 1)
            ),
            dim=1,
        )
        x = torch.cat((x, input_state), dim=1)
        x = self.head(x)
        return x
