import os
import cv2
import numpy as np

from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    def __init__(self, images_dir: str, images_size: int):
        self.ids = os.listdir(images_dir)
        self.ids.sort()
        self.images_size = images_size
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i], cv2.IMREAD_COLOR)
        image = self.get_random_crop(image, self.images_size, self.images_size)
        return self.to_tensor(image)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def get_random_crop(image, crop_height, crop_width):
        max_y = image.shape[0] - crop_height
        max_x = image.shape[1] - crop_width

        y = np.random.randint(0, max_y)
        x = np.random.randint(0, max_x)

        crop = image[y: y + crop_height, x: x + crop_width]
        return crop

    @staticmethod
    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

