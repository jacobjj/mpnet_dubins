import datetime
import os.path as osp

import numpy as np
import os
import os.path as osp
import random
import re
import torch
import csv
# import torchvision

from mpnet_dubins.model.End2end_dubins_model import End2EndMPNet
from mpnet_dubins.utils.misc import load_net_state, load_opt_state, save_state, to_var, load_seed

get_numpy = lambda x: x.data.cpu().numpy()

def generateModelPath():
    newDir = osp.join(osp.getcwd(),
                      datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    return newDir


class MPnetBase():
    """
    MPnet implementation for path planning
    """

    def __init__(self,
                 normalize,
                 denormalize,
                 encoderInputDim,
                 encoderOutputDim,
                 worldSize,
                 n_tasks=1,
                 n_memories=256,
                 memory_strength=0.5,
                 grad_steps=1,
                 AE=None,
                 MLP=None,
                 modelPath=None):
        """
        Initialize the mpnet planner
        : param IsInCollsion
        : param normalize
        : param denormalize
        : param CAE
        : param MLP
        : param worldsize
        : param n_tasks,
        : param n_memories
        : param memory_strength
        : param grad_steps : number of gradient descent steps taken for optimizing MPNet in each epoch
        : param n_epochs
        """
        self.torch_seed = np.random.randint(low=0, high=1000)
        self.np_seed = np.random.randint(low=0, high=1000)
        self.py_seed = np.random.randint(low=0, high=1000)
        torch.manual_seed(self.torch_seed)
        np.random.seed(self.np_seed)
        random.seed(self.py_seed)
        if not (AE or MLP):
            raise NotImplementedError("Add autoencoder and MLP network")
        self.normalize = normalize
        self.denormalize = denormalize
        self.worldSize = worldSize
        self.worldInputDim = len(worldSize)

        self.mpNet = End2EndMPNet(
            AE_input_size=encoderInputDim,
            AE_output_size=encoderOutputDim,
            state_size=self.worldInputDim,
            mlp_input_size=encoderOutputDim,
            n_tasks=n_tasks,
            grad_step=grad_steps,
            CAE=AE,
            MLP=MLP,
        )
        if torch.cuda.is_available():
            self.mpNet.cuda()
            self.mpNet.mlp.cuda()
            self.mpNet.encoder.cuda()

        if modelPath == None:
            modelPath = generateModelPath()

        if osp.exists(modelPath):
            self.modelPath = modelPath
        else:
            raise ValueError("Not a valid directory")

    def load_network_parameters(self, modelFile):
        """
        A method to load previously trained network parameters
        : param modelPath : location of the model parameters of the model
        """
        load_opt_state(self.mpNet, modelFile)
        load_net_state(self.mpNet, modelFile)
        torch_seed, np_seed, py_seed = load_seed(modelFile)
        torch.manual_seed(torch_seed)
        np.random.seed(np_seed)
        random.seed(py_seed)


    def save_network_state(self, fname):
        """
        A method to save states of the network
        """
        save_state(self.mpNet, self.torch_seed, self.np_seed, self.py_seed,
                   fname)

    def format_input(self, obs, inputs):
        """
        Formats the input data that needed to be fed into the network
        """
        if isinstance(inputs, np.ndarray):
            bi = torch.FloatTensor(inputs)
        else:
            bi = inputs.float()
        if isinstance(obs, np.ndarray):
            bobs = torch.FloatTensor(obs)
        else:
            bobs = obs.float()

        # Normalize observations
        # normObsVoxel = torchvision.transforms.Normalize([0.5], [1])
        # for i in range(bobs.shape[0]):
        #     bobs[i, ...] = normObsVoxel(bobs[i, ...])
        bi = self.normalize(bi, self.worldSize)
        return to_var(bobs), to_var(bi)

    def format_data(self, obs, inputs, targets):
        """
        Formats the data to be fed into the neural network
        """
        bobs, bi = self.format_input(obs, inputs)
        # Format targets
        if isinstance(targets, np.ndarray):
            bt = torch.FloatTensor(targets)
        else:
            bt = targets.float()
        bt = self.normalize(bt, self.worldSize)
        bt = to_var(bt)
        return bobs, bi, bt
