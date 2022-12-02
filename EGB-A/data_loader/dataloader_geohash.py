import math

import numpy as np

from .data_loader import InriaDataLoader
from .geohash.geohash_coding import encode


def num2deg(xtile, ytile, zoom):
    n = 2.0**zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


class InriaDataLoaderGeohash(InriaDataLoader):
    def __init__(self,
                 x_set_dir,
                 y_set_dir,
                 patch_size,
                 patch_stride,
                 batch_size,
                 shuffle=False,
                 is_train=False,
                 num_classes=2,
                 geohash_precision=None,
                 file_names=None):

        super(InriaDataLoaderGeohash, self).__init__(x_set_dir,
                                                     y_set_dir,
                                                     patch_size,
                                                     patch_stride,
                                                     batch_size,
                                                     shuffle=shuffle,
                                                     is_train=is_train,
                                                     num_classes=num_classes,
                                                     file_names=file_names)

        self.geohash_codes = []
        self.geohash_precision = geohash_precision

        if not (geohash_precision is None):
            for filename in self.file_names:

                xtile = float(filename.split('_')[3])
                ytile = float(filename.split('_')[5])
                zoom = 15
                lat, lng = num2deg(xtile, ytile, zoom)
                geohash_code_list = encode(float(lat),
                                           float(lng),
                                           precision=self.geohash_precision)

                # The geohash_array can be seen as an image
                # with size of 1*1*len.
                num_code_len = len(geohash_code_list)
                geohash_array = np.array(geohash_code_list,
                                         dtype='float32').reshape(
                                             (1, 1, num_code_len))
                self.geohash_codes.append(geohash_array)

    def get_batch_geohash(self, batch_index):
        batch_patch_idx = self.patches_index[batch_index *
                                             self.batch_size:(batch_index +
                                                              1) *
                                             self.batch_size]

        batch_geohash = []
        for patch_idx in batch_patch_idx:
            img_idx = int(patch_idx / self.patches_per_img)
            geohash_code = self.geohash_codes[img_idx]
            batch_geohash.append(geohash_code)

        #preprocess geohash code
        batch_geohash = np.array(batch_geohash, dtype='float32').copy() - 0.5

        return batch_geohash

    def get_batch_patches(self, batch_index):
        batch_image, batch_label = super().get_batch_patches(batch_index)

        if self.geohash_precision is None:
            return batch_image, batch_label
        else:
            batch_geohash = self.get_batch_geohash(batch_index)
            return [batch_image, batch_geohash], batch_label
