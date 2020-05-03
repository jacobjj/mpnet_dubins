# Save the script of the model
import torch
import numpy as np

import mpnet_dubins.model.voxel_ae as voxelNet
import mpnet_dubins.model.model as model
from mpnet_dubins.src.mpnet import MPnetBase
from mpnet_dubins.utils.misc import normalize, unnormalize


if __name__=="__main__":
    modelPath = '/root/data/grid_world_2_0_06/trained_models/mpnet_epoch_289.pkl'
    saveTorchScriptModel = '/root/data/grid_world_2_0_06/trained_models/mpnet_model_289.pt'

    network_param = {
        "normalize": normalize,
        "denormalize": unnormalize,
        "encoderInputDim": [1, 80, 80],
        "encoderOutputDim": 128,
        "worldSize": [6, 6, np.pi],
        "AE": voxelNet,
        "MLP": model.MLP,
        "modelPath": modelPath}

    mpnet_base = MPnetBase(**network_param)
    mpnet_base.load_network_parameters(modelPath)
    
    sm = torch.jit.script(mpnet_base.mpNet)
    sm.save(saveTorchScriptModel)