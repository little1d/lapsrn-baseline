import os
import torch.nn.functional
from torch.utils.data import Dataset
from taming.data.base import ImagePaths
import cv2

class SRBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None
        self.if_need_down = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example

        if self.if_need_down:
            lr_64 = cv2.resize(ex['image'], (64, 64))
            lr_128 = cv2.resize(ex['image'], (128, 128))
            lr_256 = cv2.resize(ex['image'], (256, 256))
            lr_512 = cv2.resize(ex['image'], (512, 512))
            ex['lr'] = [lr_64, lr_128, lr_256, lr_512]
        else:
            ex['lr'] = None

        return ex


class DF2KTrain(SRBase):
    def __init__(self, size, keys=None, if_down=True):
        super().__init__()
        root = "../data/DIV2K_train_HR"
        relpaths = os.listdir(root)
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys
        self.if_need_down = if_down


class DF2KValidation(SRBase):
    def __init__(self, size, keys=None, if_down=True):
        super().__init__()
        root = "../data/DIV2K_valid_HR"
        relpaths = os.listdir(root)
        paths = [os.path.join(root, relpath) for relpath in relpaths][0:10]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys
        self.if_need_down = if_down


if __name__ == "__main__":
    dataset = DF2KTrain(size=512)
