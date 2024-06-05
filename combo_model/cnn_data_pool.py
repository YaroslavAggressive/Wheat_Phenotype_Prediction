import keras
import numpy as np


class ComboDataPool(keras.utils.Sequence):

    def __init__(self, images, features, labels, batch_size: int, max_len: int = -1):
        self.batch_size = batch_size
        self.images = images[:max_len]
        self.features = features[:max_len]
        self.labels = labels[:max_len]

    def __len__(self):
        return int(np.ceil(self.images.shape[0] / self.batch_size))

    def __getitem__(self, idx):
        batch_data = [self.images[idx * self.batch_size:(idx + 1) * self.batch_size],
                      self.features[idx * self.batch_size:(idx + 1) * self.batch_size]]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_data, batch_labels
