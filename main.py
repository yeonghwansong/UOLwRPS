import argparse
import json
import random
from datetime import datetime
import numpy as np
from datasets.datasetgetter import get_dataset
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
from models import resnet, vggtf, vit
import torchvision
from validation import *
from func import *

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data/', help='path to dataset')
parser.add_argument('--dataset', default='CUB', help='type of dataset for inference', type=str,
                    choices=['CUB', 'IMAGENET', 'CARS', 'DOGS', 'AIRCRAFT'])
parser.add_argument('--base_dataset', default='N', help='type of dataset to calculate parameter w*', type=str,
                    choices=['CUB', 'IMAGENET', 'CARS', 'DOGS', 'AIRCRAFT', 'CIFAR10', 'CIFAR100', 'CIFAR100C'])
parser.add_argument('--cls_network', type=str, default = 'N',
                    choices=['ResNet50', 'EfficientNetB7', 'ViT', 'N'])
parser.add_argument('--loc_network', type=str)
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--image_size', default=256, type=int, help='resized image size')
parser.add_argument('--crop_size', default=224, type=int, help='center cropped image size')
parser.add_argument('--gpu', default="0", type=str, help='GPU id to use.')
parser.add_argument('--save_image', action="store_true")
parser.add_argument('--resnet_downscale', type=int, default=16,
                    choices=[32,16,8])
parser.add_argument('--classwise', action="store_true", help='whether using class specific parameters w*')
parser.add_argument('--caam', action="store_true", help='estimating activation map only by the norm of the features')
parser.add_argument('--sampling_ratio', type=float, default=1.0, help='sampling rate of dataset to calculate parameter w*')



import torch.backends.cudnn
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

sd=random.randint(1000,10000)
#sd=42
print("seed:",sd)
seed_everything(sd)

def main():
    args = parser.parse_args()
    if args.base_dataset=='N':
        args.base_dataset=args.dataset
    if args.base_dataset.lower()=='cubseg':
        args.base_dataset='CUB'
    if args.gpu is not None:
        print("Use GPU: {}".format(args.gpu))
        
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        args.gpu = int(args.gpu)
        args.gpu = 0
        
    classifier = None
    topk = None
    model_name='_'
    if args.cls_network == 'N':
        classifier = nn.Module()
        classifier.forward = lambda x : 0
        model_name += 'N'
        topk = None
    elif 'res' in args.cls_network.lower():
        if args.dataset.lower() == 'cub':
            import models.cubresnet
            classifier = models.cubresnet.ResNet(False)
            chpt_loader=nn.Module()
            chpt_loader.module = classifier
            chpt_loader.load_state_dict(torch.load("model_best.pth.tar",map_location='cpu'))
        else:
            classifier = torchvision.models.resnet50(True,True)
        model_name+='R50'
    elif 'efficient' in args.cls_network.lower():
        from efficientnet_pytorch import EfficientNet
        if args.dataset.lower() == "cub":
            classifier = EfficientNet.from_pretrained('efficientnet-b7', num_classes=200)
            chpt_loader = nn.Module()
            chpt_loader.module = classifier
            chpt_loader.load_state_dict(torch.load("efficientnetb7-cub200-best-epoch.pth")['state_dict'])
        else:
            classifier = EfficientNet.from_pretrained('efficientnet-b7')
        model_name+='E7'
    elif 'vit' in args.cls_network.lower():
        from models.vit_classifier import vit_small
        classifier = vit_small(8)
        checkpoint = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').state_dict()
        classifier.load_state_dict(checkpoint, strict=False)
        checkpoint = torch.load('dino_deitsmall8_linearweights.pth')['state_dict']
        classifier.load_state_dict(checkpoint, strict=False)
    else:
        classifier = nn.Module()
        classifier.forward = lambda x : 0
        model_name += args.cls_network
        topk = None
    
    
    model_name+='_'
    model = None
    if 'mocov1' in args.loc_network:
        model = resnet.resnet50(resnet_downscale=args.resnet_downscale)
        checkpoint = torch.load('moco_v1_200ep_pretrain.pth.tar', map_location="cpu")['state_dict']
        chpt_loader = nn.Module()
        chpt_loader.module = nn.Module()
        chpt_loader.module.encoder_q = model
        chpt_loader.load_state_dict(checkpoint, strict=False)
        model_name+='MOCOv1'

    elif 'mocov2' in args.loc_network:
        model = resnet.resnet50(resnet_downscale=args.resnet_downscale)
        checkpoint = torch.load('moco_r50_v2-e3b0c442.pth', map_location="cpu")['state_dict']
        model.load_state_dict(checkpoint, strict=False)
        model_name+='MOCOv2'
        
    elif 'mocov3' in args.loc_network:
        model_name+='MOCOv3'
        model = resnet.resnet50(resnet_downscale=args.resnet_downscale)
        checkpoint = torch.load('linear-1000ep.pth.tar', map_location="cpu")['state_dict']
        chpt_loader = nn.Module()
        chpt_loader.module = model
        chpt_loader.load_state_dict(checkpoint, strict=False)
        
    elif 'detco' in args.loc_network:
        model = resnet.resnet50(resnet_downscale=args.resnet_downscale)
        model_name+='DETCO'
        if 'AA' in args.loc_network:
            checkpoint = torch.load('detco_200ep_AA.pth', map_location="cpu")['state_dict']
        else:
            checkpoint = torch.load('detco_200ep.pth', map_location="cpu")['state_dict']
            model_name+='AA'
        model.load_state_dict(checkpoint, strict=False)
        
    elif 'dino' in args.loc_network:
        model_name+='DiNO'
        if 'r' in args.loc_network:
            model_name+='RES'
            model = resnet.resnet50(resnet_downscale=args.resnet_downscale)
            checkpoint = torch.load('dino_resnet50_pretrain.pth', map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
        elif 'v' in args.loc_network:
            model_name+='VIT'
            checkpoint = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').state_dict()
            model = vit.vit_small(patch_size=8)
            model.load_state_dict(checkpoint,strict=False)
            
    elif 'densecl' in args.loc_network:
        model_name+='DenseCL50'
        model = resnet.resnet50(resnet_downscale=args.resnet_downscale)
        checkpoint = torch.load('densecl_r50_imagenet_200ep.pth', map_location="cpu")['state_dict']
        model.load_state_dict(checkpoint, strict=False)
        
    elif 'byol' in args.loc_network:
        model_name+='BYOL'
        model = resnet.resnet50(resnet_downscale=args.resnet_downscale)
        checkpoint = torch.load('byol.pth.tar', map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
        
    elif 'superr50' in args.loc_network:
        model_name+='R50'
        model = resnet.resnet50(True,resnet_downscale=args.resnet_downscale)
        
    elif 'vgg' in args.loc_network:
        model_name+='VGG'
        model = vggtf.vggcamtf()
        
    elif 'simsiam' in args.loc_network:
        model_name+='SimSiam'
        model = resnet.resnet50(resnet_downscale=args.resnet_downscale)
        checkpoint = torch.load('simsiam256.pth.tar', map_location="cpu")['state_dict']
        chpt_loader = nn.Module()
        chpt_loader.module = nn.Module()
        chpt_loader.module.encoder = model
        chpt_loader.load_state_dict(checkpoint, strict=False)
        
    elif 'supcon' in args.loc_network:
        model_name+='SupCon'
        model = resnet.resnet50(resnet_downscale=args.resnet_downscale)
        checkpoint = torch.load('supcon.pth', map_location="cpu")['model']
        chpt_loader = nn.Module()
        chpt_loader.module = nn.Module()
        chpt_loader.module.encoder = model
        chpt_loader.load_state_dict(checkpoint, strict=False)
        
    elif 'ccam' in args.loc_network:
        model = resnet.resnet50(resnet_downscale=args.resnet_downscale,ccam=True)
        checkpoint = torch.load('current_epoch_5.pth', map_location="cpu")['state_dict']
        chpt_loader = nn.Module()
        chpt_loader.backbone = model
        chpt_loader.load_state_dict(checkpoint, strict=False)
        model_name+='CCAM'
        
    elif 'daol' in args.loc_network:
        model = resnet.resnet50(resnet_downscale=args.resnet_downscale)
        checkpoint = torch.load('dawsol_ilsvrc_res.tar', map_location="cpu")['state_dict_extractor']
        model.load_state_dict(checkpoint, strict=False)
        model_name+='DAOL'
        
    if model is None:
        print("NOT IMPLEMENTED")
        return
        
    networks = [model.cuda(args.gpu), classifier.cuda(args.gpu)]
    
    exp_info = args.dataset + (f'_{args.base_dataset}' if args.dataset != args.base_dataset else "") + model_name + f'_{args.resnet_downscale}'

    if args.sampling_ratio < 1.:
        exp_info = exp_info + "_" + str(args.sampling_ratio)
    exp_info = exp_info + ('_class' if args.classwise else '') +'_{}'.format(datetime.now().strftime("%m-%d_%H-%M-%S"))

    formatted_print('Total Number of Workers:', args.workers)
    formatted_print('Batch Size:', args.batch_size)
    formatted_print('Image Size:', args.image_size)
    formatted_print('Crop Size:', args.crop_size)
    formatted_print('Classification:', args.cls_network)
    formatted_print('Localization:', args.loc_network)
    
    model.eval()
    
    cudnn.benchmark = True
    makedirs('./results')
    args.res_dir = os.path.join('./results', exp_info)
    formatted_print('Result DIR:', args.res_dir)
    makedirs(args.res_dir)
    with open(os.path.join(args.res_dir, "args.json"), 'wt') as f:
        json.dump(args.__dict__, f, indent=2)
    if args.dataset.lower() == 'cubseg' or args.dataset.lower() == 'openimages':
        val_dataset = get_dataset(args.dataset, args)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                num_workers=args.workers, pin_memory=True)
        out = segmentation(val_loader, networks, args, saveimgs=args.save_image)
        with open(os.path.join(args.res_dir, "resulttest.json"), 'wt') as f:
            json.dump(out, f, indent=2)
    else:
        if args.dataset.lower() == 'cub' or args.dataset.lower() == 'imagenet':
            val_dataset = get_dataset(args.dataset, args, 'val')
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                    num_workers=args.workers, pin_memory=True)
            outval = localization(val_loader, networks, args, saveimgs=False, topk = topk)
            with open(os.path.join(args.res_dir, "resultval.json"), 'wt') as f:
                json.dump(outval, f, indent=2)
            test_dataset = get_dataset(args.dataset, args, 'test')
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                      num_workers=args.workers, pin_memory=True)
            outtest = localization(test_loader, networks, args, act_threshold = [outval["BestThreshold"]],saveimgs=args.save_image, topk = topk)
            with open(os.path.join(args.res_dir, "resulttest.json"), 'wt') as f:
                json.dump(outtest, f, indent=2)
        else:
            val_dataset = get_dataset(args.dataset, args,'test')
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                    num_workers=args.workers, pin_memory=True)
            out = localization(val_loader, networks, args, saveimgs=args.save_image, topk = topk)
            with open(os.path.join(args.res_dir, "resulttest.json"), 'wt') as f:
                json.dump(out, f, indent=2)

if __name__ == '__main__':
    main()
