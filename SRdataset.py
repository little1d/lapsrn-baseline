import glob
import random
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as tf
from PIL import Image


def transform(img, settype):
    # Resize
    if settype == "train":
        # Random horizontal flipping
        if random.random() > 0.5:
            img = tf.hflip(img)

        if random.random() > 0.5:
            img = tf.vflip(img)

        # Random rotation
        rotations = [0, 90, 180, 270]
        pick_rotation = rotations[random.randint(0, 3)]
        img = tf.rotate(img, pick_rotation)

        # Transform to tensor
        img_lr = tf.to_tensor(transforms.Resize((32, 32), Image.BICUBIC)(img))
        img_2x = tf.to_tensor(transforms.Resize((64, 64), Image.BICUBIC)(img))
        img_4x = tf.to_tensor(img)
    else:
        img_lr = tf.to_tensor(transforms.Resize((32, 32), Image.BICUBIC)(img))
        img_2x = tf.to_tensor(transforms.Resize((64, 64), Image.BICUBIC)(img))
        img_4x = tf.to_tensor(img)

    return img_lr, img_2x, img_4x


# 自定义数据集类，用于加载和预测处理图像数据，继承父类dataset
class SRdataset(Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, settype):
        """Initialization"""
        # 定义后缀名字典
        extension_dict = {'train': '*.jpg', 'test': '*.png', 'validation': '*.bmp'}
        file_extension = extension_dict.get(settype, '*.jpg')
        self.list_ids = glob.glob('dataset/{}/*'.format(settype) + file_extension)  # 获取所有图像文件的index
        self.true_len = len(self.list_ids)  # 实际加载的图像数量
        self.settype = settype  # 数据集类型
        self.patch_size = 128  # 训练图像的裁剪大小
        self.eps = 1e-3

    def __len__(self):
        """Denotes the total number of samples"""
        # 对于train dataset 重复使用图像以数据增强
        if self.settype == "train":
            # return 64000
            return 300
        else:
            return len(self.list_ids)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        if self.settype == 'train':
            id = self.list_ids[int(self.true_len * index / self.__len__())]
        else:
            id = self.list_ids[index]

        # Load data and get label
        img = Image.open(id)
        img = img.convert('YCbCr')
        img = img.getchannel(0)

        if self.settype == 'train':
            resize_factor = random.uniform(0.5, 1)

            if img.size[0] < img.size[1]:
                if img.size[0] * resize_factor < self.patch_size:
                    resize_factor = self.patch_size / img.size[0] + self.eps
            else:
                if img.size[1] * resize_factor < self.patch_size:
                    resize_factor = self.patch_size / img.size[1] + self.eps

            img = img.resize((int(img.size[0] * resize_factor), int(img.size[1] * resize_factor)), Image.BICUBIC)
            img = transforms.RandomCrop((self.patch_size, self.patch_size))(img)
        else:
            img = img.resize((self.patch_size, self.patch_size), Image.BICUBIC)

        return transform(img, self.settype)
