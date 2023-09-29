from __future__ import annotations
import torch.utils.data as data
import scipy.io
import numpy as np
import os
from datasets.cub200 import pil_loader


class Cars(data.Dataset):

    def __init__(self, data_dir, args, is_train=True, transform=None):
        """
        Args:
            mat_anno (string): Path to the MATLAB annotation file.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.transform = transform
        self.data_dir = os.path.join(data_dir, 'stanfordCar')
        self.annoroot_dir = os.path.join(self.data_dir, 'devkit')

        if is_train:
            self.image_dir = os.path.join(self.data_dir, 'cars_train')
            self.anno_dir = os.path.join(self.annoroot_dir, 'cars_train_annos.mat')
            self.size_dir = os.path.join(self.annoroot_dir, 'cars_train_size.txt')
        else:
            self.image_dir = os.path.join(self.data_dir, 'cars_test')
            self.anno_dir = os.path.join(self.annoroot_dir, 'cars_test_annos_withlabels.mat')
            self.size_dir = os.path.join(self.annoroot_dir, 'cars_test_size.txt')

        # x1 y1 x2 y2 label filename
        self.annotations = scipy.io.loadmat(self.anno_dir)
        self.annotations = self.annotations['annotations'][0]
        self.load_bboxes(args.image_size, args.crop_size)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # x1 y1 x2 y2 label filename
        fileid = self.annotations[idx][-1].item()
        img_dir = os.path.join(self.image_dir, fileid)

        target = self.annotations[idx][-2].item()
        bbox = self.bboxes[idx]

        img = pil_loader(img_dir)

        if self.transform:
            img = self.transform(img)

        return img, idx, fileid, target, bbox

    def load_bboxes(self, image_size, crop_size):
        size_file = open(self.size_dir, 'r')
        sizes = size_file.readlines()
        size_idx = 0
        self.bboxes = [[] for i in range(len(self.annotations))]
        for idx in range(len(self.annotations)):
            w, h = int(sizes[size_idx].strip().split(' ')[0]), int(sizes[size_idx].strip().split(' ')[1])
            size_idx += 1
            x1, y1, x2, y2 = [self.annotations[idx][0].item(), self.annotations[idx][1].item(),
                              self.annotations[idx][2].item(), self.annotations[idx][3].item()]

            x_scale = image_size / w
            y_scale = image_size / h
            max_=crop_size - 1

            x1_new = np.clip(np.round(x1 * x_scale - (image_size - crop_size) * 0.5), 0, max_)
            y1_new = np.clip(np.round(y1 * y_scale - (image_size - crop_size) * 0.5), 0, max_)
            x2_new = np.clip(np.round(x2 * x_scale - (image_size - crop_size) * 0.5), x1_new, max_)
            y2_new = np.clip(np.round(y2 * y_scale - (image_size - crop_size) * 0.5), y1_new, max_)

            self.bboxes[idx] = np.array([[x1_new, y1_new, x2_new, y2_new],], dtype = np.float32).tobytes()
