import fnmatch
import os
import random

import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa
from keras.utils import Sequence, to_categorical
from skimage.io import imread


def rotation(image, label, nb_classes):

    seq = iaa.Sequential([
        iaa.Affine(
            rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects segmaps)
        #iaa.Resize((0.75, 1.25),interpolation='nearest')
    ])
    segmap = ia.SegmentationMapOnImage(label.copy(),
                                       shape=label.shape,
                                       nb_classes=nb_classes)

    seq_det = seq.to_deterministic()

    image_rotation = seq_det.augment_image(image.copy())
    segmap_aug = seq_det.augment_segmentation_maps(segmap)

    label_rotation = segmap_aug.get_arr_int()

    reduction_pixels = int(0.15 * label.shape[0])
    start_i = reduction_pixels
    stop_i = label.shape[0] - reduction_pixels
    return image_rotation[start_i:stop_i, start_i:stop_i, :], label_rotation[
        start_i:stop_i, start_i:stop_i]


class InriaDataLoader(Sequence):
    def __init__(self,
                 x_set_dir,
                 y_set_dir,
                 patch_size,
                 patch_stride,
                 batch_size,
                 shuffle=False,
                 is_train=False,
                 num_classes=2,
                 is_multi_scale=False,
                 file_names=None):
        if file_names is None:
            self.file_names = [
                file_name for file_name in os.listdir(x_set_dir)
                if fnmatch.fnmatch(file_name, '*.tif') or fnmatch.fnmatch(file_name, '*.png')
            ]
        else:
            self.file_names = file_names

        self.images_filenames = [
            os.path.join(x_set_dir, item) for item in self.file_names
        ]
        self.labels_filenames = [
            os.path.join(y_set_dir, item) for item in self.file_names
        ]

        img = imread(self.images_filenames[0], plugin='gdal').astype(np.uint8)
        self.image_height = img.shape[0]
        self.image_width = img.shape[1]
        self.num_bands = img.shape[2]

        self.is_train = is_train
        self.is_multi_scale = is_multi_scale
        if self.is_train:
            self.patch_height = int(patch_size[0] * 2.14)
            self.patch_width = int(patch_size[1] * 2.14)

            self.sample_height = patch_size[0]
            self.sample_width = patch_size[1]
        else:
            self.patch_height = patch_size[0]
            self.patch_width = patch_size[1]

        self.patch_height_stride = patch_stride[0]
        self.patch_width_stride = patch_stride[1]

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.num_classes = num_classes
        self.patches_index = np.arange(0, self.num_patches)
        self.reset()

    @property
    def patch_rows_per_img(self):
        return int((self.image_height - self.patch_height) /
                   self.patch_height_stride) + 1

    @property
    def patch_cols_per_img(self):
        return int((self.image_width - self.patch_width) /
                   self.patch_width_stride) + 1

    @property
    def patches_per_img(self):
        return self.patch_rows_per_img * self.patch_cols_per_img

    @property
    def num_imgs(self):
        return len(self.images_filenames)

    @property
    def num_patches(self):
        return self.patches_per_img * self.num_imgs

    def _get_patch(self, filenames, patch_idx):
        img_idx = int(patch_idx / self.patches_per_img)
        img_patch_idx = patch_idx % self.patches_per_img
        row_idx = int(img_patch_idx / self.patch_cols_per_img)
        col_idx = img_patch_idx % self.patch_cols_per_img

        img = imread(filenames[img_idx], plugin='gdal').astype(np.uint8)
        if len(img.shape) > 2:
            patch_image = img[row_idx * self.patch_height_stride:row_idx *
                              self.patch_height_stride +
                              self.patch_height, col_idx *
                              self.patch_width_stride:col_idx *
                              self.patch_width_stride +
                              self.patch_width, :].copy()
        else:
            patch_image = img[row_idx * self.patch_height_stride:row_idx *
                              self.patch_height_stride +
                              self.patch_height, col_idx *
                              self.patch_width_stride:col_idx *
                              self.patch_width_stride +
                              self.patch_width].copy()
        return patch_image

    def get_patch(self, filenames, patch_idx):
        return self._get_patch(filenames, patch_idx)

    def data_augmentation(self, image, label):

        crop_size = random.randint(int(0.8 * self.sample_height),
                                   int(1.2 * self.sample_height))

        start_h = random.randint(0, image.shape[0] - int(1.42 * crop_size) - 2)
        start_w = random.randint(0, image.shape[1] - int(1.42 * crop_size) - 2)

        

        image_crop = image[start_h:start_h +
                      int(1.42 * crop_size), start_w:start_w +
                      int(1.42 * crop_size)].copy()
        label_crop = label[start_h:start_h +
                      int(1.42 * crop_size), start_w:start_w +
                      int(1.42 * crop_size)].copy()


        seq = iaa.Sequential([
            iaa.Affine(shear=(-4, 4), rotate=(
                0, 360)),  # rotate by -45 to 45 degrees (affects segmaps)    
        ])
        segmap = ia.SegmentationMapOnImage(label_crop,
                                           shape=label_crop.shape,
                                           nb_classes=self.num_classes)

        seq_det = seq.to_deterministic()

        image_rotation = seq_det.augment_image(image_crop)
        segmap_aug = seq_det.augment_segmentation_maps(segmap)

        label_rotation = segmap_aug.get_arr_int()

        reduction_pixels = int(0.15 * label_rotation.shape[0])
        start_i = reduction_pixels
        stop_i = label_crop.shape[0] - reduction_pixels
        image_crop = image_rotation[start_i:stop_i, start_i:stop_i, :]
        label_crop = label_rotation[start_i:stop_i, start_i:stop_i]

        seq = iaa.Sequential([
            iaa.Resize(
                {
                    "height": self.sample_height,
                    "width": self.sample_width
                },
                interpolation='nearest'),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Sometimes(0.8, iaa.HistogramEqualization()),
            iaa.Sometimes(
                0.8, iaa.CoarseDropout((0.0, 0.05),
                                       size_percent=(0.02, 0.25))),
            iaa.AddToHueAndSaturation((-20, 20), per_channel=True),
        ])
        segmap = ia.SegmentationMapOnImage(label_crop,
                                           shape=label_crop.shape,
                                           nb_classes=self.num_classes)

        seq_det = seq.to_deterministic()

        image_aug = seq_det.augment_image(image_crop)
        segmap_aug = seq_det.augment_segmentation_maps(segmap)

        label_aug = segmap_aug.get_arr_int()
        
        


        return image_aug, label_aug

    def preprocess_input(self, img):
        img = img.astype('float32')
        img -= 128.0
        img /= 128.0
        return img

    def preprocess_label(self, label):
        label = label.reshape((-1, ))
        label[label == 255] = 1
        label_one_hot = to_categorical(label, num_classes=3)
        #label_one_hot = to_categorical(label, num_classes=self.num_classes)

        label_one_hot = label_one_hot.astype('float32')

        return label_one_hot

    def get_batch_patches(self, batch_index):
        batch_patch_idx = self.patches_index[batch_index *
                                             self.batch_size:(batch_index +
                                                              1) *
                                             self.batch_size]

        batch_image = []
        batch_label = []
        for patch_idx in batch_patch_idx:
            image = self.get_patch(self.images_filenames, patch_idx)
            label = self.get_patch(self.labels_filenames, patch_idx)

            label[label > 0] = 1
            if self.is_train:
                image, label = self.data_augmentation(image, label)

            image = self.preprocess_input(image)
            label = self.preprocess_label(label)

            batch_image.append(image)
            batch_label.append(label)

        batch_image = np.array(batch_image)
        batch_label = np.array(batch_label)

        return batch_image, batch_label

    def __getitem__(self, index):
        """Gets batch at position `index`.
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch
        """
        batch_image, batch_label = self.get_batch_patches(index)

        return batch_image, batch_label

    def __len__(self):
        """Number of batch in the Sequence.
        # Returns
            The number of batches in the Sequence.
        """
        return int(self.patches_index.shape[0] / self.batch_size)

    def select_valid_sample(self):
        raw_patches_index = list(np.arange(0, self.num_patches))
        valid_patches_index = []
        for patch_idx in raw_patches_index:
            label = self._get_patch(self.labels_filenames, patch_idx)
            if not np.all(label == 0):
                valid_patches_index.append(patch_idx)
        self.patches_index = np.array(valid_patches_index)

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.patches_index)

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        self.reset()
