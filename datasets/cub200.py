import os
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

def pil_loader(path, gray=False):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if gray:
            return img.convert('L')
        else:
            return img.convert('RGB')

        
class Cub2011(Dataset):
    base_folder = '../data/CUB/'

    def __init__(self, root, args, split='train', transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.split = split
        self.data = []
        self.label = {}
        self.bboxes = {}
        self.cls=(args.cls_network!='N')
        if self.cls:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            if 'eff' in args.cls_network:
                self.cls_transform = transforms.Compose([
                    transforms.Resize((685, 685)),
                    transforms.CenterCrop((600, 600)),
                    transforms.ToTensor(),
                    normalize,
                    ])
            else:
                self.cls_transform = transforms.Compose([
                    transforms.Resize((448, 448)),
                    transforms.ToTensor(),
                    normalize,
                    ])
        
        with open(os.path.join("datasets/metadata",'CUB',split,'class_labels.txt'),"rt") as f:
            for line in f.readlines():
                image_id, label = line.strip('\n').split(',')
                self.data.append(image_id)
                self.label[image_id]=int(label)
        sizes={}
        boxes={}
        with open(os.path.join("datasets/metadata",'CUB',split,'image_sizes.txt'),"rt") as f:
            for line in f.readlines():
                image_id, xs, ys = line.strip('\n').split(',')
                x, y = int(xs), int(ys)
                if image_id in sizes:
                    sizes[image_id].append((x,y))
                else:
                    sizes[image_id] = [(x,y)]
        with open(os.path.join("datasets/metadata",'CUB',split,'localization.txt'),"rt") as f:
            for line in f.readlines():
                image_id, x0s, x1s, y0s, y1s = line.strip('\n').split(',')
                x0, x1, y0, y1 = int(x0s), int(x1s), int(y0s), int(y1s)
                if image_id in boxes:
                    boxes[image_id].append((x0, x1, y0, y1))
                else:
                    boxes[image_id] = [(x0, x1, y0, y1)]
            
        if self.split=='train':
            return
        for image_id in self.data:
            resized_bbox=[]
            for box in boxes[image_id]:
                x1, y1, x2, y2 = box
                
                x_scale = args.image_size / sizes[image_id][0][0]
                y_scale = args.image_size / sizes[image_id][0][1]
                max_ = args.crop_size - 1
                
                x1_new = np.clip(np.round(x1 * x_scale - (args.image_size - args.crop_size) * 0.5), 0, max_)
                y1_new = np.clip(np.round(y1 * y_scale - (args.image_size - args.crop_size) * 0.5), 0, max_)
                x2_new = np.clip(np.round(x2 * x_scale - (args.image_size - args.crop_size) * 0.5), x1_new, max_)
                y2_new = np.clip(np.round(y2 * y_scale - (args.image_size - args.crop_size) * 0.5), y1_new, max_)

                resized_bbox.append([x1_new, y1_new, x2_new, y2_new])
            self.bboxes[image_id] = np.array(resized_bbox,dtype=np.float32).tobytes()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_id = self.data[idx]
        image = self.loader(os.path.join(self.root, image_id))
        if self.transform is not None:
            img = self.transform(image)
        if self.split=='train':
            return img, idx, image_id, self.label[image_id], 0.
        if self.cls:
            cls_img = self.cls_transform(image)
            return img, idx, image_id, self.label[image_id], self.bboxes[image_id], cls_img
        return img, idx, image_id.split('/')[-1].split('.')[0], self.label[image_id], self.bboxes[image_id]


def load_mask_image(file_path, resize_size, center_crop):
    """
    Args:
        file_path: string.
        resize_size: tuple of ints (height, width)
    Returns:
        mask: numpy.ndarray(dtype=numpy.float32, shape=(height, width))
    """
    mask = np.float32(cv2.imread(file_path, cv2.IMREAD_GRAYSCALE))/255.
    mask = cv2.resize(mask, resize_size, interpolation=cv2.INTER_LINEAR)
    crop=int(round((resize_size[0] - center_crop)*0.5))
    return mask[crop:crop+center_crop,crop:crop+center_crop]

class CUBwithSEG(Cub2011):
    def __init__(self, root, args, split='train', transform=None):
        super().__init__(root, args, split, transform)
        self.mask_folder = self.root / 'segmentations'
        self.image_size = args.image_size
        self.crop_size = args.crop_size
            
    def __getitem__(self, idx):
        image_id = self.data[idx]
        target=self.label[image_id]
        img = self.loader(os.path.join(self.base_folder, image_id))
        if self.transform is not None:
            img = self.transform(img)
        if self.split == 'train':
            return img, idx, image_id, target, 0.
        mask = load_mask_image(os.path.join(self.mask_folder, image_id)[:-3]+'png', (self.image_size, self.image_size), self.crop_size)
        mask=(mask>0.5).astype(np.uint8)
        return img, idx, image_id.split('/')[-1].split('.')[0], target, mask
