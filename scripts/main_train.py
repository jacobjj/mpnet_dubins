import numpy as np
import argparse
import torch

# import src.Model.AE.CAE as CAE_2d
import mpnet_dubins.model.voxel_ae as voxelNet
import mpnet_dubins.model.model as model


from mpnet_dubins.utils.misc import normalize, unnormalize
from mpnet_dubins.src.train import MPnetTrain


def train(args):
    denormalize = unnormalize
    MLP = model.MLP
    network_parameters = {
        'normalize': normalize,
        'denormalize': denormalize,
        'encoderInputDim': [1, 80, 80],
        'encoderOutputDim': 128,
        # 'worldSize': [27, 27, np.pi],
        'worldSize' : [ 6, 6, np.pi],
        'AE': voxelNet,
        'MLP': MLP,
        'modelPath': args.file,
    }
    
    trainNetwork = MPnetTrain(
        load_dataset=None,
        n_epochs=300,
        batchSize=256,
        opt=torch.optim.Adam,
        learning_rate=3e-4,
        **network_parameters,
    )
    # trainNetwork.set_model_train_epoch(999)

    trainNetwork.train(numEnvsTrain=150000,
                       numEnvsTest=1000,
                       numPaths=1,
                       trainDataPath='/root/data/mymap/training_data',
                       testDataPath='/root/data/mymap/training_data')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
