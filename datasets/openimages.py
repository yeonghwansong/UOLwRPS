import os
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2

def load_mask_image(file_path, resize_size, center_crop):
    """
    Args:
        file_path: string.
        resize_size: tuple of ints (height, width)
    Returns:
        mask: numpy.ndarray(dtype=numpy.float32, shape=(height, width))
    """
    mask = np.float32(cv2.imread(file_path, cv2.IMREAD_GRAYSCALE))
    mask = cv2.resize(mask, resize_size, interpolation=cv2.INTER_NEAREST)
    crop=int(round((resize_size[0] - center_crop)*0.5))
    return mask[crop:crop+center_crop,crop:crop+center_crop]

def pil_loader(path, gray=False):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if gray:
            return img.convert('L')
        else:
            return img.convert('RGB')

class OpenImages(Dataset):
    def __init__(self, root, args, split='train', transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.split = split
        self.image_size = args.image_size
        self.crop_size = args.crop_size
        self.data = []
        self.label = {}
        self.mask_paths = {}
        self.ignore_path = {}
        self.mba = args.mba
        
        with open(os.path.join("/home/syh/cl/usm/datasets/metadata",'OpenImages',split,'class_labels.txt'),"rt") as f:
            for line in f.readlines():
                image_id, label = line.strip('\n').split(',')
                self.data.append(image_id)
                self.label[image_id]=int(label)
        with open(os.path.join("/home/syh/cl/usm/datasets/metadata",'OpenImages',split,'localization.txt'),"rt") as f:
            for line in f.readlines():
                image_id, mask_path, ignore_path = line.strip('\n').split(',')
                if image_id in self.mask_paths:
                    self.mask_paths[image_id].append(mask_path)
                    assert (len(ignore_path) == 0)
                else:
                    self.mask_paths[image_id] = [mask_path]
                    self.ignore_path[image_id] = ignore_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_id = self.data[idx]
        target=self.label[image_id]
        img = self.loader(os.path.join(self.root, image_id))
        if self.transform is not None:
            img = self.transform(img)
        if self.split == 'train':
            return img, idx, image_id, target, 0.
        mask_all_instances = []
        for mask_path in self.mask_paths[image_id]:
            mask_file = os.path.join(self.root, mask_path)
            mask = load_mask_image(mask_file, (self.image_size, self.image_size), self.crop_size)
            mask_all_instances.append(mask > 0.5)
        mask_all_instances = np.stack(mask_all_instances, axis=0).any(axis=0)

        ignore_file = os.path.join(self.root, self.ignore_path[image_id])
        ignore_box_mask = load_mask_image(ignore_file, (self.image_size, self.image_size), self.crop_size)
        ignore_box_mask = ignore_box_mask > 0.5

        ignore_mask = np.logical_and(ignore_box_mask,
                                    np.logical_not(mask_all_instances))

        if np.logical_and(ignore_mask, mask_all_instances).any():
            raise RuntimeError("Ignore and foreground masks intersect.")

        return img, idx, image_id.split('/')[-1].split('.')[0], target, (mask_all_instances.astype(np.uint8) + 255 * ignore_mask.astype(np.uint8))
        