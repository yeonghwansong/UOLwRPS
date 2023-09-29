# from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from datasets.cub200 import CUBwithSEG, Cub2011
import os
import torchvision.transforms as transforms
from datasets.dogs import Dogs
from datasets.cars import Cars
from datasets.aircraft import AIRCRAFT
from datasets.ilsvrc import ILSVRC
from datasets.openimages import OpenImages

def get_dataset(dataset, args, split='test'):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transforms_val = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.CenterCrop((args.crop_size, args.crop_size)),
        transforms.ToTensor(),
        normalize,
    ])
    
    if dataset.lower() == 'cub':
        print('USE CUB DATASET')
        val_dataset = Cub2011(os.path.join(args.data_dir, args.dataset), args,
                              transform=transforms_val, split=split)
        
    elif dataset.lower() == 'cubseg':
        print('USE CUB DATASET')
        val_dataset = CUBwithSEG(os.path.join(args.data_dir, args.dataset), args,
                              transform=transforms_val, split='test')
        
    elif dataset.lower() == 'imagenet':
        print("USE IMAGENET")
        val_dataset = ILSVRC(os.path.join(args.data_dir, 'ILSVRC'), args,
                              transform=transforms_val, split=split)
        
    elif dataset.lower() == 'openimages':
        print("USE OpenImages")
        val_dataset = OpenImages(os.path.join(args.data_dir, 'OpenImages'), args,
                              transform=transforms_val, split='test')
        
    elif dataset.lower() == 'cifar10':
        print('NOT IMPLEMENTED DATASET :', dataset)
        exit(-3)
        
    elif dataset.lower() == 'cifar100':
        print('NOT IMPLEMENTED DATASET :', dataset)
        exit(-3)
        
    elif dataset.lower() == 'cars':
        print('USE CARS DATASET')
        val_dataset = Cars(args.data_dir, args, False, transforms_val)

    elif dataset.lower() == 'dogs':
        print('USE DOGS DATASET')
        val_dataset = Dogs(args.data_dir, args, transform=transforms_val, train=False)
        
    elif dataset.lower() == 'aircraft':
        print('USE AIRCRAFT DATASET')
        val_dataset = AIRCRAFT(args.data_dir, args, transform=transforms_val, train='val')

    else:
        print('NOT IMPLEMENTED DATASET :', dataset)
        exit(-3)

    return val_dataset
