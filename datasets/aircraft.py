import torch.utils.data as data
import numpy as np
import os
from PIL import Image


class AIRCRAFT(data.Dataset):

    IMAGE_FOLDER = "AIRCRAFT/data/images"
    DATA_FOLDER = "AIRCRAFT/data"
    IMG_EXTENSIONS = '.jpg'
    # CLASS_FOLDER = "ImageSets/Main"

    # def __init__(self, data_dir, is_train, transform=None, with_id=False):
    def __init__(self, root, args, train='train', transform=None):
        """
        Args:
            mat_anno (string): Path to the MATLAB annotation file.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.transform = transform
        self.train = train

        self.ground_truth = dict(
            image_list=[],
            image_sizes={},
            gt_bboxes={},
            resized_gt_bboxes={},
        )

        if train == 'train':
            file_path = os.path.join(self.root, self.DATA_FOLDER, 'images_train_size.txt')
        elif train == 'val':
            file_path = os.path.join(self.root, self.DATA_FOLDER, 'images_val_size.txt')
        elif train == 'test':
            file_path = os.path.join(self.root, self.DATA_FOLDER, 'images_test_size.txt')

        bbox_file_path = os.path.join(self.root, self.DATA_FOLDER, 'images_box.txt')

        # get metadata about image_list / image_size
        with open(file_path) as f:
            for line in f:
                img_name, w, h = line.split()
                self.ground_truth['image_list'].append(img_name)
                self.ground_truth['image_sizes'][img_name] = (float(w), float(h))

        # get metadata about image gt_bboxes
        with open(bbox_file_path) as f:
            for line in f:
                img_name, x1, y1, x2, y2 = line.split()
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                if not img_name in self.ground_truth['image_sizes'].keys():
                    continue
                self.ground_truth['gt_bboxes'][img_name] = (x1, y1, x2, y2)
                w, h = self.ground_truth['image_sizes'][img_name]
                
                x_scale = args.image_size / w
                y_scale = args.image_size / h
                max_=args.crop_size - 1

                x1_new = np.clip(np.round(x1 * x_scale - (args.image_size - args.crop_size) * 0.5), 0, max_)
                y1_new = np.clip(np.round(y1 * y_scale - (args.image_size - args.crop_size) * 0.5), 0, max_)
                x2_new = np.clip(np.round(x2 * x_scale - (args.image_size - args.crop_size) * 0.5), x1_new, max_)
                y2_new = np.clip(np.round(y2 * y_scale - (args.image_size - args.crop_size) * 0.5), y1_new, max_)
                
                self.ground_truth['resized_gt_bboxes'][img_name] = np.array([[x1_new, y1_new, x2_new, y2_new],], dtype = np.float32).tobytes()

    def __len__(self):
        return len(self.ground_truth['image_list'])

    def __getitem__(self, index):

        img_id = self.ground_truth['image_list'][index]
        bbox = self.ground_truth['resized_gt_bboxes'][img_id]
        img_path = os.path.join(self.root, self.IMAGE_FOLDER, img_id + self.IMG_EXTENSIONS)

        img = Image.open(img_path).convert('RGB')

        # TODO : Not yet implemented about target
        target = 0

        if self.transform is not None:
            img = self.transform(img)
            
        return img, index, img_id, target, bbox


if __name__ == '__main__':
    test = AIRCRAFT(root='../data', with_id=True)
    img, target, id = test.__getitem__(0)
    # print(test.ground_truth['gt_bboxes'])
