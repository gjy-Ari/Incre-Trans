from keras.applications.nasnet import preprocess_input

from .dataloader_geohash_boost import InriaDataLoaderGeohashBoost


class InriaDataLoaderGeohashBoostNASNet(InriaDataLoaderGeohashBoost):
    def preprocess_input(self, img):
        img = img.astype('float32')
        img = preprocess_input(img)
        return img
