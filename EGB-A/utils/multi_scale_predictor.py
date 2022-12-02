import itertools

import keras
import numpy as np
from keras import backend as K
from keras.layers import AveragePooling2D, Input
from keras.models import Model

from .predictor import SegmentationPrediction


class MultiscalePredictor():
    def __init__(self,
                 img_array,
                 preprocess_input,
                 model_func,
                 weights_path,
                 number_of_class,
                 img_size=1024,
                 geohash_array=None,
                 geohash_precision=None):

        self.weights_path = weights_path
        self.model_func = model_func
        self.preprocess_input = preprocess_input
        self.number_of_class = number_of_class
        self.img_array = img_array.copy()
        self.geohash_precision = geohash_precision
        self.is_geohash = False

        if not (geohash_precision is None):
            self.is_geohash = True
            self.geohash_array = geohash_array.copy()

        self.img_size = img_size
        self.start_list = [
            i for i in range(0, 4096 - img_size, int(img_size / 2))
        ]
        self.start_list.append(4096 - img_size)

    def set_geohash(self):
        if not (self.geohash_precision is None):
            self.input_geohash_shape = (1, 1, self.geohash_precision)
            self.geohash_input = Input(shape=self.input_geohash_shape)

    def build_input(self, img_array, img_input, scale):
        img_list = []
        #for i, j in zip(self.slice_row_start, self.slice_col_start):
        for i, j in itertools.product(self.start_list, repeat=2):
            img_list.append(img_array[int(scale * i):int(scale * i) +
                                      int(scale * self.img_size),
                                      int(scale * j):int(scale * j) +
                                      int(scale * self.img_size), :].copy())

        if self.is_geohash:
            self.set_geohash()
            network_input = [img_input, self.geohash_input]

            input_array = [[item, self.geohash_array] for item in img_list]
        else:
            network_input = [img_input]
            input_array = [item for item in img_list]
        return network_input, input_array

    def load_model(self, network_input):
        model = self.model_func(network_input,
                                number_of_class=self.number_of_class)
        model = Model(inputs=network_input, outputs=model)
        for layer in model.layers:
            layer.trainable = False
        model.load_weights(self.weights_path, by_name=True)
        return model

    def predict_scale_1(self):
        scale = 1
        scale_1_input_shape = (self.img_size, self.img_size, 3)
        keras.backend.clear_session()
        img_input = Input(shape=scale_1_input_shape)
        network_input, input_array = self.build_input(self.img_array,
                                                      img_input,
                                                      scale=scale)
        model = self.load_model(network_input)

        predict_list = []
        for item in input_array:
            result = SegmentationPrediction(
                model, item, self.preprocess_input, self.img_size,
                self.number_of_class, self.is_geohash).predict_8_orientaion()
            predict_list.append(result)
        keras.backend.clear_session()
        tile_mask = np.zeros((4096, 4096, 2), dtype='float32')
        for (i, j), item in zip(itertools.product(self.start_list, repeat=2),
                                predict_list):
            tile_mask[i:i + self.img_size, j:j + self.img_size, :] += item

        return tile_mask

    def predict_multi_scale(self):
        img_prob_list = []
        keras.backend.clear_session()
        img_prob_list.append(self.predict_scale_1())
        keras.backend.clear_session()

        prob_sum = np.array(img_prob_list).sum(axis=0)
        return prob_sum
