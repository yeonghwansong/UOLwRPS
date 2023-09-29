# https://github.com/zrsmithson/Stanford-dogs/blob/master/data/stanford_dogs_data.py
from __future__ import print_function
from PIL import Image
from os.path import join
import os
import scipy.io

import torch.utils.data as data
from torchvision.datasets.utils import list_dir
import numpy as np

def pil_loader(path, gray=False):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if gray:
            return img.convert('L')
        else:
            return img.convert('RGB')
        
class Dogs(data.Dataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        cropped (bool, optional): If true, the images will be cropped into the bounding box specified
            in the annotations
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset tar files from the internet and
            puts it in root directory. If the tar files are already downloaded, they are not
            downloaded again.
    """
    folder = 'stanfordDogs'
    download_url_prefix = 'http://vision.stanford.edu/aditya86/ImageNetDogs'

    def __init__(self,
                 root,
                 args,
                 train=True,
                 transform=None):

        self.root = join(os.path.expanduser(root), self.folder)
        self.train = train
        self.transform = transform

        split = self.load_split()

        self.images_folder = join(self.root, 'Images')
        self.annotations_folder = join(self.root, 'Annotation')
        self._breeds = list_dir(self.images_folder)

        self._breed_annotations = [(annotation + '.jpg', annotation, idx, self.get_boxes(join(self.annotations_folder, annotation),args.image_size,args.crop_size))
                                    for annotation, idx in split]
        #for i in range(len(self._flat_breed_annotations)):
        #    if(len(self._breed_annotations[i])>1):
        #        print(self._breed_annotations[i])
        #        print(self._flat_breed_annotations[i])

        self.classes = self.get_classes()

    def __len__(self):
        return len(self._breed_annotations)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        
        image_name, image_id, target_class, bbox = self._breed_annotations[index]
        image_path = join(self.images_folder, image_name)
        image = pil_loader(image_path)

        if self.transform:
            image = self.transform(image)

        return image, index, image_id.split("/")[-1], target_class, bbox[0]

    @staticmethod
    def get_boxes(path, image_size, crop_size):
        import xml.etree.ElementTree
        e = xml.etree.ElementTree.parse(path).getroot()
        boxes = []
        sizes = []
        ori = []
        for size in e.iter('size'):
            sizes.append([int(size.find('width').text),
                          int(size.find('height').text)])
        for objs in e.iter('object'):
            x1,y1,x2,y2=int(objs.find('bndbox').find('xmin').text),int(objs.find('bndbox').find('ymin').text),int(objs.find('bndbox').find('xmax').text),int(objs.find('bndbox').find('ymax').text)
            ori.append([x1, y1, x2, y2])
            
            image_width, image_height = sizes[0]

            x_scale = image_size / image_width
            y_scale = image_size / image_height
            max_ = crop_size - 1
            x1_new = np.clip(np.round(x1 * x_scale - (image_size - crop_size) * 0.5), 0, max_)
            y1_new = np.clip(np.round(y1 * y_scale - (image_size - crop_size) * 0.5), 0, max_)
            x2_new = np.clip(np.round(x2 * x_scale - (image_size - crop_size) * 0.5), x1_new, max_)
            y2_new = np.clip(np.round(y2 * y_scale - (image_size - crop_size) * 0.5), y1_new, max_)
            boxes.append([x1_new, y1_new, x2_new, y2_new])
        
        return (np.array(boxes,dtype=np.float32).tobytes(), ori, size)

    def load_split(self):
        if self.train:
            split = scipy.io.loadmat(join(self.root, 'train_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(join(self.root, 'train_list.mat'))['labels']
        else:
            split = scipy.io.loadmat(join(self.root, 'test_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(join(self.root, 'test_list.mat'))['labels']

        split = [item[0][0] for item in split]
        labels = [item[0]-1 for item in labels]
        return list(zip(split, labels))

    def stats(self):
        counts = {}
        for index in range(len(self._breed_annotations)):
            image_name, image_id, target_class, bbox  = self._breed_annotations[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print("%d samples spanning %d classes (avg %f per class)"%(len(self._breed_annotations), len(counts.keys()), float(len(self._breed_annotations))/float(len(counts.keys()))))

        return counts

    def get_classes(self):
        return ["Chihuaha",
                        "Japanese Spaniel",
                        "Maltese Dog",
                        "Pekinese",
                        "Shih-Tzu",
                        "Blenheim Spaniel",
                        "Papillon",
                        "Toy Terrier",
                        "Rhodesian Ridgeback",
                        "Afghan Hound",
                        "Basset Hound",
                        "Beagle",
                        "Bloodhound",
                        "Bluetick",
                        "Black-and-tan Coonhound",
                        "Walker Hound",
                        "English Foxhound",
                        "Redbone",
                        "Borzoi",
                        "Irish Wolfhound",
                        "Italian Greyhound",
                        "Whippet",
                        "Ibizian Hound",
                        "Norwegian Elkhound",
                        "Otterhound",
                        "Saluki",
                        "Scottish Deerhound",
                        "Weimaraner",
                        "Staffordshire Bullterrier",
                        "American Staffordshire Terrier",
                        "Bedlington Terrier",
                        "Border Terrier",
                        "Kerry Blue Terrier",
                        "Irish Terrier",
                        "Norfolk Terrier",
                        "Norwich Terrier",
                        "Yorkshire Terrier",
                        "Wirehaired Fox Terrier",
                        "Lakeland Terrier",
                        "Sealyham Terrier",
                        "Airedale",
                        "Cairn",
                        "Australian Terrier",
                        "Dandi Dinmont",
                        "Boston Bull",
                        "Miniature Schnauzer",
                        "Giant Schnauzer",
                        "Standard Schnauzer",
                        "Scotch Terrier",
                        "Tibetan Terrier",
                        "Silky Terrier",
                        "Soft-coated Wheaten Terrier",
                        "West Highland White Terrier",
                        "Lhasa",
                        "Flat-coated Retriever",
                        "Curly-coater Retriever",
                        "Golden Retriever",
                        "Labrador Retriever",
                        "Chesapeake Bay Retriever",
                        "German Short-haired Pointer",
                        "Vizsla",
                        "English Setter",
                        "Irish Setter",
                        "Gordon Setter",
                        "Brittany",
                        "Clumber",
                        "English Springer Spaniel",
                        "Welsh Springer Spaniel",
                        "Cocker Spaniel",
                        "Sussex Spaniel",
                        "Irish Water Spaniel",
                        "Kuvasz",
                        "Schipperke",
                        "Groenendael",
                        "Malinois",
                        "Briard",
                        "Kelpie",
                        "Komondor",
                        "Old English Sheepdog",
                        "Shetland Sheepdog",
                        "Collie",
                        "Border Collie",
                        "Bouvier des Flandres",
                        "Rottweiler",
                        "German Shepard",
                        "Doberman",
                        "Miniature Pinscher",
                        "Greater Swiss Mountain Dog",
                        "Bernese Mountain Dog",
                        "Appenzeller",
                        "EntleBucher",
                        "Boxer",
                        "Bull Mastiff",
                        "Tibetan Mastiff",
                        "French Bulldog",
                        "Great Dane",
                        "Saint Bernard",
                        "Eskimo Dog",
                        "Malamute",
                        "Siberian Husky",
                        "Affenpinscher",
                        "Basenji",
                        "Pug",
                        "Leonberg",
                        "Newfoundland",
                        "Great Pyrenees",
                        "Samoyed",
                        "Pomeranian",
                        "Chow",
                        "Keeshond",
                        "Brabancon Griffon",
                        "Pembroke",
                        "Cardigan",
                        "Toy Poodle",
                        "Miniature Poodle",
                        "Standard Poodle",
                        "Mexican Hairless",
                        "Dingo",
                        "Dhole",
                        "African Hunting Dog"]
