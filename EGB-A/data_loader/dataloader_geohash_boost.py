import numpy as np

from .dataloader_geohash import InriaDataLoaderGeohash


class InriaDataLoaderGeohashBoost(InriaDataLoaderGeohash):
    def get_batch_patches(self, batch_index):
        batch_image_batch_geohash_list, batch_label = super(
        ).get_batch_patches(batch_index)

        batch_patch_idx = self.patches_index[batch_index *
                                             self.batch_size:(batch_index +
                                                              1) *
                                             self.batch_size]

        batch_patch_x = []
        batch_patch_y = []

        for patch_idx in batch_patch_idx:

            # EGB and EGB-A: no regional differentiation
            patch_x = 1
            patch_y = 1

            # GeoBoost: divide different regions by file name
            # img_idx = int(patch_idx / self.patches_per_img)
            # file_name = self.file_names[img_idx]
            # patch_x = file_name.split('_')[3]
            # patch_y = file_name.split('_')[5]

            batch_patch_x.append(patch_x)
            batch_patch_y.append(patch_y)

        batch_patch_x = np.array(batch_patch_x, dtype='float32')
        batch_patch_y = np.array(batch_patch_y, dtype='float32')

        if self.geohash_precision is None:
            batch_image_batch_geohash_list = [batch_image_batch_geohash_list]

        batch_image_batch_geohash_list.append(batch_patch_x)
        batch_image_batch_geohash_list.append(batch_patch_y)

        return batch_image_batch_geohash_list, batch_label
