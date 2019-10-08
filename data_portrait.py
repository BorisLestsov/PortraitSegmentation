import torch
import numpy as np
import PIL.Image as Image
import os

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class SegDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        super(SegDataset, self).__init__()

        self.root = root
        self.transform = transform
        self.img_dir  = os.path.join(root, 'images')
        self.mask_dir = os.path.join(root, 'masks')
        self.samples = os.listdir(self.img_dir)
        self.samples = list(filter(lambda x: os.path.splitext(x)[1] in IMG_EXTENSIONS, self.samples))
        self.samples = [(os.path.join(self.img_dir, i), os.path.join(self.mask_dir, i[:-4]+'.png')) for i in self.samples]


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        img_path, mask_path = self.samples[index]

        img = np.asarray(Image.open(img_path).convert('RGB'))
        mask = np.asarray(Image.open(mask_path))

        if self.transform is not None:
            img, mask = self.transform((img, mask))

        return img, mask

    def __len__(self):
        return len(self.samples)


