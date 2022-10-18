import numpy as np
from .image_processing import ImageProcessing


class SegmentationPrediction():
    def __init__(self,
                 seg_model,
                 seg_inputs,
                 preprocess_input,
                 img_size,
                 number_of_class,
                 is_geohash=False):
        self.model = seg_model
        self.is_geohash = is_geohash
        if self.is_geohash:
            self.img = seg_inputs[0]
            self.geohash_array = seg_inputs[1]
        else:
            self.img = seg_inputs

        self.img = self.img.copy().astype('float32')
        self.preprocess_input = preprocess_input
        self.img = self.preprocess_input(self.img)

        self.img_size = img_size
        self.number_of_class = number_of_class

        self.img_prob_list = []

    def predict_img(self, input_image):
        input_image_array = np.array([input_image], dtype='float32')
        if self.is_geohash:
            geohash_array = np.array([self.geohash_array.copy()],
                                     dtype='float32')
            input_list = [input_image_array, geohash_array]
        else:
            input_list = input_image_array
        result = self.model.predict(input_list, batch_size=1)

        img_prob = result[0]
        img_result = img_prob.reshape(self.img_size, self.img_size,
                                      self.number_of_class)
        return img_result

    def predict_raw(self):
        predict_prob = self.predict_img(self.img.copy())
        self.img_prob_list.append(predict_prob)

    def predict_vertical_flip(self):
        img_post = ImageProcessing(self.img).vertical_flip().img
        predict_prob = self.predict_img(img_post)
        recover_predict_prob = ImageProcessing(
            predict_prob).vertical_flip().img
        self.img_prob_list.append(recover_predict_prob)

    def predict_horizontal_flip(self):
        img_post = ImageProcessing(self.img).horizontal_flip().img
        predict_prob = self.predict_img(img_post)
        recover_predict_prob = ImageProcessing(
            predict_prob).horizontal_flip().img
        self.img_prob_list.append(recover_predict_prob)

    def predict_rot90(self):
        img_post = ImageProcessing(self.img).rot90().img
        predict_prob = self.predict_img(img_post)
        recover_predict_prob = ImageProcessing(predict_prob).rot270().img
        self.img_prob_list.append(recover_predict_prob)

    def predict_rot270(self):
        img_post = ImageProcessing(self.img).rot270().img
        predict_prob = self.predict_img(img_post)
        recover_predict_prob = ImageProcessing(predict_prob).rot90().img
        self.img_prob_list.append(recover_predict_prob)

    def predict_rot180(self):
        img_post = ImageProcessing(self.img).rot180().img
        predict_prob = self.predict_img(img_post)
        recover_predict_prob = ImageProcessing(predict_prob).rot180().img
        self.img_prob_list.append(recover_predict_prob)

    def predict_rot90_vertical_flip(self):
        img_post = ImageProcessing(self.img).rot90().vertical_flip().img
        predict_prob = self.predict_img(img_post)
        recover_predict_prob = ImageProcessing(
            predict_prob).vertical_flip().rot270().img
        self.img_prob_list.append(recover_predict_prob)

    def predict_rot270_vertical_flip(self):
        img_post = ImageProcessing(self.img).rot270().vertical_flip().img
        predict_prob = self.predict_img(img_post)
        recover_predict_prob = ImageProcessing(
            predict_prob).vertical_flip().rot90().img
        self.img_prob_list.append(recover_predict_prob)

    def predict_8_orientaion(self):
        self.predict_raw()
        self.predict_horizontal_flip()
        self.predict_vertical_flip()
        self.predict_rot90()
        self.predict_rot180()
        self.predict_rot270()
        self.predict_rot90_vertical_flip()
        self.predict_rot270_vertical_flip()
        prob_sum = np.array(self.img_prob_list).sum(axis=0)
        return prob_sum
