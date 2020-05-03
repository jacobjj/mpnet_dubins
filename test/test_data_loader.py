from torch.utils.data import DataLoader
from mpnet_dubins.utils.data_loader import DubinsDataset


if __name__ == "__main__":
    trainDataFileName = '/root/data/'
    train_ds = DubinsDataset(trainDataFileName, 10)
    train_dl = DataLoader(train_ds, shuffle=True, num_workers=5, batch_size=1, drop_last=True)