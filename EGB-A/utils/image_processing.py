import numpy as np


class ImageProcessing():
    def __init__(self, img_array):
        self.img = img_array.copy()

    def horizontal_flip(self):
        self.img = self.img[:, ::-1]
        return self

    def vertical_flip(self):
        self.img = self.img[::-1, :]
        return self

    def rot90(self):
        self.img = np.rot90(self.img, k=1)
        return self

    def rot180(self):
        self.img = np.rot90(self.img, k=2)
        return self

    def rot270(self):
        self.img = np.rot90(self.img, k=3)
        return self
