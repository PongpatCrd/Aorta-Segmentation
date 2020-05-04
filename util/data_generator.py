import cv2
import numpy as np
import keras
import configs as cfgs
from util.data_util import (process_data, process_roi, crop_size, data_augmentation, 
    divide_image, pad_size, find_best_piece, flat_data)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, images, rois, batch_size=4, norm=False, shuffle=False, odd_even_rand=False, data_aug=False):
        self.images = images
        self.rois = rois
        self.batch_size = batch_size
        self.norm = norm
        self.shuffle = shuffle
        self.odd_even_rand = odd_even_rand
        self.data_aug = data_aug
        self.on_epoch_end()

    def __len__(self):
        if self.odd_even_rand:
            return int(int(len(self.images)/2) / self.batch_size)
        else:
            return int(len(self.images) / self.batch_size)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        x, y = self.__data_generation(indexes)

        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.images))

        if self.odd_even_rand:
            self.indexes = self._odd_even_rand(self.indexes)
        
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        x = np.zeros((len(indexes), cfgs.train_size_ny, cfgs.train_size_nx, 1), dtype=np.float32)
        y = np.zeros((len(indexes), cfgs.train_size_ny, cfgs.train_size_nx, cfgs.n_classes), dtype=np.bool)
        
        for i in range(len(indexes)):

            tmp_x = cv2.imread(self.images[indexes[i]], 0).astype(np.float32)
            tmp_y = cv2.imread(self.rois[indexes[i]], 0).astype(np.bool)

            tmp_x = divide_image(tmp_x)
            tmp_y = divide_image(tmp_y)

            best_piece = find_best_piece(tmp_y)

            tmp_x = tmp_x[best_piece, ]
            tmp_y = tmp_y[best_piece, ]

            if self.data_aug:
                tmp_x, tmp_y = data_augmentation(tmp_x, tmp_y)

            if self.norm:
                tmp_x = process_data(tmp_x)

            x[i, 0:cfgs.train_size_ny, 0:cfgs.train_size_nx, :] = flat_data(tmp_x, 1)
            y[i, 0:cfgs.train_size_ny, 0:cfgs.train_size_nx, :] = process_roi(tmp_y)

        return x, y

    def _odd_even_rand(self, indexes):
        # odd, even = 0, 1
        take = np.random.randint(0, 2)
        if take == 0:
            return indexes[(indexes % 2) != 0]
        else:
            return indexes[(indexes % 2) == 0]
