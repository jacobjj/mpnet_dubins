import sys
import os.path as osp

import numpy as np
import csv
import pandas as pd

import torch
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader

from mpnet_dubins.utils.data_loader import DubinsDataset
from mpnet_dubins.src.mpnet import MPnetBase

# TODO: Change the code to center the robot
def CenterRobot(costmap, pixel_ind):
    costmap_data = costmap.get_data()
    costmap_dim = costmap_data.shape
    full_obs = np.ones((costmap_dim[0] * 2, costmap_dim[1] * 2))
    x_0, y_0 = costmap_dim[1] - pixel_ind[1], costmap_dim[0] - pixel_ind[0]
    full_obs[x_0:x_0 + costmap_dim[1], y_0:y_0 +
             costmap_dim[0]] = costmap_data / 254
    full_obs = full_obs[::3, ::3]
    full_obs = torch.Tensor(full_obs).unsqueeze(0)
    return full_obs

get_numpy = lambda x: x.data.cpu().numpy()


class MPnetTrain(MPnetBase):
    """
    Sets up the training of MPnet
    """

    def __init__(self,
                 load_dataset,
                 n_epochs=1000,
                 batchSize=256,
                 learning_rate=1e-2,
                 opt=None,
                 **kwargs):
        """
        Initialize the MPnet trainer
        """
        super().__init__(**kwargs)
        self.train_loss = []
        self.test_loss = []
        self.start_epoch = 0
        self.n_epochs = n_epochs
        self.batchSize = batchSize
        self.load_dataset = load_dataset
        if opt is None:
            opt = torch.optim.Adagrad
        self.mpNet.set_opt(opt, lr=learning_rate)
        # self.mpNet.set_opt(torch.optim.SGD, lr=learning_rate)

    def set_model_train_epoch(self, epoch):
        fileLoc = osp.join(self.modelPath, 'mpnet_epoch_{}.pkl'.format(epoch))
        self.load_network_parameters(fileLoc)
        data = pd.read_csv(osp.join(self.modelPath, 'progress.csv'))
        self.test_loss = list(data['test_loss'][:epoch])
        self.train_loss = list(data['train_loss'][:epoch])
        self.start_epoch = epoch + 1

    def train(self, numEnvsTrain, numEnvsTest, numPaths, trainDataPath,
              testDataPath):
        """
        A method to train the network with given data
        """
        print('Loading data...')
        # obs, inputs, targets = self.load_dataset(N=numEnvsTrain,
        #                                          NP=numPaths,
        #                                          folder_loc=trainDataPath)
        # trainObs, trainInput, trainTarget = self.format_data(
        #     obs, inputs, targets)

        # obs_test, inputs_test, targets_test = self.load_dataset(
        #     N=numEnvsTest, NP=1, folder_loc=testDataPath)
        # testObs, testInput, testTarget = self.format_data(
        #     obs_test, inputs_test, targets_test)

        train_ds = DubinsDataset(trainDataPath, numEnvsTrain*numPaths)
        train_dl = DataLoader(train_ds, shuffle=True, num_workers = 5, batch_size = self.batchSize, drop_last=True)

        test_ds = DubinsDataset(testDataPath, numEnvsTest*numPaths)
        testObs, testInput, testTarget = test_ds[:int(numEnvsTest*numPaths/2)]
        testObs, testInput, testTarget = self.format_data(
            testObs, testInput, testTarget)

        # Setting the learning rate scheduler
        # scheduler = step_decay_schedule(initial_lr=3e-4,
        #                                 decay_factor=0.75,
        #                                 step_size=100)
        # scheduler = cyclic_schedule(maximum_lr=3e-4,
        #                             minimum_lr=3e-5,
        #                             step_size=100)

        # lr_range = np.linspace(-6, 0.1, self.n_epochs / 2)

        # Train the Models
        print('Training...')

        for epoch in range(self.start_epoch, self.n_epochs):
            batch_loss = 0
            grad_norm = []
            self.mpNet.train()
            train_loss_i = 0
            for batch in train_dl:
                bobs, bi, bt = batch
                bobs, bi, bt = self.format_data(bobs, bi, bt)
                # Run gradient descent
                train_loss_i += self.mpNet.fit(bobs, bi, bt)
                # grad_norm.append(self.mpNet(bobs, bi, bt))
            train_loss_i /=len(train_dl)

            with torch.no_grad():
                # test loss
                network_output = self.mpNet(testInput, testObs)
                test_loss_i = self.mpNet.loss(
                    network_output,
                    testTarget
                    ).sum(dim=1).mean()
                test_loss_i = get_numpy(test_loss_i)

            print('Epoch {} - train loss: {}'.format(epoch, train_loss_i))
            print('Epoch {} - test loss: {}'.format(epoch, test_loss_i))

            self.train_loss.append(train_loss_i)
            self.test_loss.append(test_loss_i)

            # Save the models
            if (epoch + 1) % 10 == 0:
                model_file = 'mpnet_epoch_%d.pkl' % (epoch)
                self.save_network_state(osp.join(self.modelPath, model_file))

            results = {
                'test_loss': self.test_loss,
                'train_loss': self.train_loss
            }

            # Record training and testing loss
            with open(osp.join(self.modelPath, 'progress.csv'), 'w') as f:
                fieldnames = results.keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                row_data = [
                    dict(
                        zip(fieldnames,
                            [results[key][i] for key in fieldnames]))
                    for i in range(len(results['train_loss']))
                ]
                for row in row_data:
                    writer.writerow(row)

        # Do a test sampling using the sample code
        # s = 304299
        # env = RandomMiniEnv(draw_new_turn_on_reset=False,
        #                 seed=s,
        #                 goal_spat_dist=0.05)
        # observation = env.reset()
        # costmap = observation.costmap
        # start = observation.pose
        # start = torch.tensor(start).float().reshape(1,-1)
        # goal = env._env._state.original_path[-1]
        # goal = torch.tensor(goal).float().reshape(1,-1)

        # center_obs = CenterRobot(costmap, costmap.world_to_pixel(start[0,:2].numpy()))
        # network_input = torch.cat((start,goal), dim=1)
        # tobs, tInput = self.format_input(center_obs.unsqueeze(0), network_input)
        # temp = self.mpNet(tInput, tobs).data.cpu()
        # temp = self.denormalize(temp.squeeze(), self.worldSize)

        # traj = np.load('data/dubinsCar/traj/traj_{}.npy'.format(s))
        # print('Network Output : {}, trajectory value: {}'.format(temp, traj[1,:]))
