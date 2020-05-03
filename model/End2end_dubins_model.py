import torch.nn as nn
import torch
from torch.distributions import MultivariateNormal

import numpy as np
import copy
from torch.nn.utils import clip_grad_norm_

get_numpy = lambda x: x.data.cpu().numpy()


def normalize_cost(z):
    """
    A function to wrap around -1 and 1
    """
    return (z + 1) % 2 - 1

class End2EndMPNet(nn.Module):
    """ A python class defining the components of MPnet"""

    def __init__(
            self,
            AE_input_size,
            AE_output_size,
            state_size,
            mlp_input_size,
            n_tasks,
            grad_step,
            CAE,
            MLP,
    ):
        """
        : param total_input_size :
        : param AE_input_size :
        : param mlp_input_size :
        : param output_size :
        : param AEtype :
        : param n_tasks :
        : param n_memories :
        : param memory_strength :
        : param grad_step :
        : param CAE :
        : param MLP :
        """
        super(End2EndMPNet, self).__init__()
        self.encoder = CAE.Encoder(AE_output_size, state_size, AE_input_size)
        self.mlp = MLP(mlp_input_size, 3)
        self.mse = nn.MSELoss()
        self.set_opt(torch.optim.Adam, lr=3e-4)
        self.lambda1 = 0.1

        self.num_seen = np.zeros(n_tasks).astype(int)
        self.grad_step = grad_step
        self.AE_input_size = AE_input_size
        # Remove any gradient set during initialization
        self.zero_grad()

    @torch.jit.ignore
    def set_opt(self, opt, lr=1e-2, momentum=None):
        # edit: can change optimizer type when setting
        if momentum is None:
            self.opt = opt(list(self.encoder.parameters()) +
                           list(self.mlp.parameters()),
                           lr=lr)
        else:
            self.opt = opt(list(self.encoder.parameters()) +
                           list(self.mlp.parameters()),
                           lr=lr,
                           momentum=momentum)

    def forward(self, x, obs):
        """
        Forward step of MPnet
        : param obs : input to encoder
        : param x : input to mlp
        """
        c = self.encoder(obs, x)
        return self.mlp(c)

    @torch.jit.ignore
    def get_path_length(self, startNode, endNode):
        """
        A function to generate dubins path object
        :param startNode : A tuple indicating the start position
        :param endNode : A tuple indicating goal position
        : returns (scalar) : path length of the shortest dubins curve.
        """
        raise NotImplementedError

    @torch.jit.ignore
    def dubins_path_loss(self, x, y):
        """
        A function that estimates the dubins curve distance from x to y.
        """
        raise NotImplementedError
        
    @torch.jit.ignore
    def loss(self, pred, truth):
        # try:
        #     contractive_loss = self.encoder.get_contractive_loss()
        # except AttributeError:
        #     return self.mse(pred, truth)
        # NOTE: This cost function is designed for r2d cars and need to change to
        # be compatible with other methods
        loss = (pred-truth)
        try:
            loss[:, 0] = loss[:, 0].clone()**2
            loss[:, 1] = loss[:, 1].clone()**2
            loss[:, 2] = normalize_cost(loss[:, 2].clone())**2
        except IndexError:
            import pdb;pdb.set_trace()
        return loss

    @torch.jit.ignore
    def loss_with_regularize(self, pred, truth):
        """
        Loss with regularization included.
        """
        loss = 0
        for i in range(3):
            regression_loss = torch.sum(
                self.loss(pred[:, i * 3:(i + 1) * 3],
                          truth[:, i * 3:(i + 1) * 3]),
                dim=1,
            )
            loss += torch.mean(regression_loss)
        return loss

    @torch.jit.ignore
    def fit(self, obs, x, y):
        """
        Updates the network weights to best fit the given data.
        :param obs: the voxel representation of the obstacle space
        :param x: the state of the robot
        :param y: the next state of the robot
        NOTE: It is safer to call nn.Module.zero_grad() rather than optim.zero_grad(). If the encoder and decoder network has different optim functions, then this takes care for setting gradients of both model to zero.
        """
        with torch.autograd.set_detect_anomaly(True):
            network_output = self.__call__(x, obs)
            self.zero_grad()
            # l1_loss = self.lambda1 * torch.norm(final_param, 1)
            loss = self.loss(network_output, y)
            # print(loss[:5,:].data.cpu().numpy())
            loss = loss.sum(dim=1).mean()
            loss.backward()
            self.opt.step()
        return loss.data.cpu().numpy()


    @torch.jit.ignore
    def fit_distribution(self, obs, x, y):
        """
        Updates the network weights to generate the best distribution, that maximizes the dubins distance from the sampled point to the dubins curve.
        :param obs: the voxel representation of the obstacle space
        :param x: the state of the robot
        :param y: the next state of the robot
        """
        y_hat, log_prob_y_hat = self.sample(obs, x)
        distance = self.dubins_path_loss(y_hat, y)
        loss = torch.mean(log_prob_y_hat * distance)
        loss.backward()
        # grad_norm = clip_grad_norm_(self.encoder.parameters(),1.5)
        grad_norm = clip_grad_norm_(self.mlp.parameters(), 1.5)
        self.opt.step()
        self.zero_grad()
        return grad_norm

    @torch.jit.ignore
    def sample(self, obs, x):
        """
        A function that returns the sampled point along with its log probability
        """
        mean = self.forward(x, obs).cpu()
        m = MultivariateNormal(mean, self.covar_matrix)
        next_state = m.sample()
        log_prob = m.log_prob(next_state)
        return next_state, log_prob
