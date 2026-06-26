import numpy as np
import torch

from config import Config


class Data:
    def __init__(self, train_file: str, val_file: str, config: Config):
        self.train_file = train_file
        self.val_file = val_file
        self.config = config

        self.train_data = np.memmap(self.train_file, dtype=np.uint16, mode="r")
        self.val_data = np.memmap(self.val_file, dtype=np.uint16, mode="r")
        # tokens = self.enc.encode(text)
        # data = torch.tensor(tokens, dtype=torch.long)

        # n = int(config.train_split * len(data))
        # self.train_data = data[:n]
        # self.val_data = data[n:]

        """
    def get_batch_raw(self, split: str):
        if split == "train":
            data = self.train_data
        else:
            data = self.val_data

        max_start = len(data) - self.config.block_size - 1 # don't start training chunk after this otherwise index??? overflow

        ix = torch.randint(max_start, (self.config.batch_size,))

        x = torch.stack([data[i : i + self.config.block_size] for i in ix]) # x.shape = batch_size * block_size
        y = torch.stack([data[i + 1 : i + self.config.block_size + 1] for i in ix]) # y.shape = batch_size * block_size (sequence length)

        x = x.to(self.config.device)
        y = y.to(self.config.device)

        return x, y
        """

    def get_batch(self, split: str):

        if split == "train":
            data = self.train_data
        else:
            data = self.val_data

        max_start = len(data) - self.config.block_size - 1

        ix = torch.randint(max_start, (self.config.batch_size,))

        x = torch.stack([torch.from_numpy(data[i : i + self.config.block_size]) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + self.config.block_size + 1]) for i in ix])

        x = x.to(self.config.device, dtype=torch.long)
        y = y.to(self.config.device, dtype=torch.long)

        return x, y
