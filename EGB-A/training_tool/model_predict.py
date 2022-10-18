import numpy as np
from skimage.io import imsave, imread

from keras.models import Model
import os


from .build_boost_model import (build_boost_model,
                                   build_inputs)


class Boost_Model():
    def __init__(self,
                 segmentation_model,
                 weights_path_list,
                 number_of_class,
                 preprocess_input,
                 input_size,
                 boost_lr,
                 geo_range_list):
        self.segmentation_model = segmentation_model
        self.weights_path_list = weights_path_list
        self.number_of_class = number_of_class
        self.preprocess_input = preprocess_input
        self.input_size = input_size
        self.boost_lr = boost_lr
        self.geo_range_list = geo_range_list

        self.load_boost_model()

    def load_boost_model(self
                                  ):
        boost_lr_list = [self.boost_lr] * (len(self.weights_path_list)-1)+[1]
        network_inputs = build_inputs(number_of_class=self.number_of_class,
                                      train_input_size=(
                                          self.input_size, self.input_size),
                                      geohash_precision=None)

        model, current_base_model = build_boost_model(
            self.segmentation_model, self.weights_path_list, self.geo_range_list, boost_lr_list, network_inputs, self.number_of_class)
        self.model = Model(inputs=network_inputs, outputs=model)
        print("Predicting using one GPU..")

    def get_input(self, file_name, input_folder):
        image_path = os.path.join(input_folder, file_name)
        img = imread(image_path, plugin='gdal').astype('float32')
        img = self.preprocess_input(img)
        img_list = [img]

        patch_x = 1
        patch_y = 1

        #patch_x = file_name.split('_')[3]
        #patch_y = file_name.split('_')[5]

        batch_patch_x = [patch_x]
        batch_patch_y = [patch_y]

        batch_patch_x = np.array(batch_patch_x, dtype='float32')
        batch_patch_y = np.array(batch_patch_y, dtype='float32')

        batch_image_and_x_y_list = [img_list]
        batch_image_and_x_y_list.append(batch_patch_x)
        batch_image_and_x_y_list.append(batch_patch_y)

        return batch_image_and_x_y_list

    def predict_and_save_prediction(self, input_folder, output_folder, file_name):

        # input preprocessing
        input_list = self.get_input(file_name, input_folder)

        # predict
        result = self.model.predict(input_list, batch_size=1)
        img_prob = result[0]
        img_result = img_prob.reshape(self.input_size, self.input_size,
                                      self.number_of_class)

        # save_probability
        imsave(os.path.join(output_folder, file_name), img_result[:,:,1])

        # # convert probability to image
        # img_max = np.argmax(img_result, axis=-1)
        # tile_label = np.zeros(
        #     (self.input_size, self.input_size), dtype=np.uint8)
        # tile_label[img_max == 1] = 255
        # imsave(os.path.join(output_folder, file_name[:-3]+'tif'), tile_label)

    
    def predict_accuracy(self, input_img_folder, file_name):

        # input preprocessing
        input_list = self.get_input(file_name, input_img_folder)

        # predict
        result = self.model.predict(input_list, batch_size=1)
        img_prob = result[0]
        img_result = img_prob.reshape(self.input_size*self.input_size,
                                      self.number_of_class)

        # convert probability to image
        img_max = np.argmax(img_result, axis=-1)
        img_max = img_max.reshape((self.input_size*self.input_size,))
        return img_max

