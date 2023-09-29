import math
from typing import Iterator
from tqdm import trange
import tqdm
from torch.nn import functional as F
import torch.nn
import torch.nn.parallel
import torch.optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.cub200 import Cub2011
from datasets.ilsvrc import ILSVRC
from datasets.openimages import OpenImages
from datasets.dogs import Dogs
from datasets.cars import Cars
from datasets.aircraft import AIRCRAFT
from datasets.cifar import CIFAR10, CIFAR100
import torch.utils.data
from func import *

def load_param(args, loc, cls=None):
    makedirs('./parameter_cache')
    parampath=f'./parameter_cache/{args.loc_network}_{args.base_dataset}_{args.resnet_downscale}' +\
            ("_class" if args.classwise else "") +\
            (("_"+str(args.sampling_ratio)) if args.sampling_ratio < 1.0 else "")
    if os.path.exists(parampath):
        return torch.load(parampath,map_location='cpu').cuda(args.gpu)
        
    
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        transforms_train = transforms.Compose([transforms.Resize((224, 224)),
                                            #transforms.CenterCrop((224, 224)),
                                            transforms.ToTensor(),
                                            normalize])
        
        nclass=0
        if args.base_dataset.lower() == 'cub':
            train_dataset = Cub2011(os.path.join(args.data_dir, 'CUB'), args,
                            transform=transforms_train, split='train')
            nclass=200
        elif args.base_dataset.lower() == 'dogs':
            train_dataset = Dogs(args.data_dir, args,
                            transform=transforms_train, train=True)
        elif args.base_dataset.lower() == 'aircraft':
            train_dataset = AIRCRAFT(args.data_dir, args,
                            transform=transforms_train, train='train')
        elif args.base_dataset.lower() == 'cars':
            train_dataset = Cars(args.data_dir, args,
                            transform=transforms_train, is_train=True)
        elif args.base_dataset.lower() == 'imagenet':
            train_dataset = ILSVRC(os.path.join(args.data_dir, 'ILSVRC'), args,
                            transform=transforms_train, split='train')
            nclass=1000
        elif args.base_dataset.lower() == 'openimages':
            train_dataset = OpenImages(os.path.join(args.data_dir, 'OpenImages'), args,
                            transform=transforms_train, split='train')
            nclass=100
        elif args.base_dataset.lower() == 'cifar10':
            train_dataset = CIFAR10('', transform=transforms_train, train=True)
            nclass=10
        elif args.base_dataset.lower() == 'cifar100':
            train_dataset = CIFAR100('', transform=transforms_train, train=True)
            nclass=100
        elif args.base_dataset.lower() == 'cifar100c':
            train_dataset = CIFAR100('', transform=transforms_train, train=True, coarse = True)
            nclass=20
            
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.workers, pin_memory=True)
        
        if args.classwise:
            v = None
            cnt = torch.zeros((nclass,1)).cuda(args.gpu)
            sample_per_class = math.floor(len(train_dataset) * args.sampling_ratio / nclass)
            for i, data in enumerate(tqdm.tqdm(train_loader)):
                if i >= len(train_loader) * args.sampling_ratio:
                    if (cnt >= sample_per_class).all().item():
                        break
                    
                target = data[3]
                if not args.cls_network == 'N':
                    cls_img = data[-1].cuda(args.gpu)
                    target = cls(cls_img).argmax(1)
                
                img=data[0].cuda(args.gpu)
                feat = loc(img)
                N,C,H,W = feat.shape
                feat = feat.reshape(N,C,-1)
                
                if v is None:
                    v=torch.zeros((nclass,C)).cuda(args.gpu)
                    u=torch.zeros((nclass,C)).cuda(args.gpu)
                    
                v_ = feat.mean(-1)
                u_ = F.normalize(feat,p=2,dim=1).mean(-1)
                
                for n in range(N):
                    if sample_per_class < 10:
                        if cnt[target[n]] > sample_per_class:
                            continue
                    v[target[n]] = (v_[n] + v[target[n]] * cnt[target[n]]) / (cnt[target[n]] + 1)
                    u[target[n]] = (u_[n] + u[target[n]] * cnt[target[n]]) / (cnt[target[n]] + 1)
                    cnt[target[n]] += 1
        else:
            v = None
            cnt = 0
            train_iter = iter(train_loader)
            for i in tqdm.tqdm(range(math.ceil(math.floor(len(train_dataset) * args.sampling_ratio) / args.batch_size))):
                try:
                    data = next(train_iter)
                except StopIteration:
                    continue
                img=data[0].cuda(args.gpu)
                if cnt+img.shape[0] > math.floor(len(train_dataset) * args.sampling_ratio):
                    img = img[:math.floor(len(train_dataset) * args.sampling_ratio) - cnt]
                feat = loc(img)
                
                N,C,H,W = feat.shape
                feat = feat.reshape(N,C,-1)
                
                if v is None:
                    v = feat.mean(-1).mean(0,keepdim=True)
                    u = F.normalize(feat,p=2,dim=1).mean(-1).mean(0,keepdim=True)
                else:
                    v = (feat.mean(-1).sum(0,keepdim=True) + v * cnt) / (cnt + img.shape[0])
                    u = (F.normalize(feat,p=2,dim=1).mean(-1).sum(0,keepdim=True) + u * cnt) / (cnt + img.shape[0])
                cnt += img.shape[0]
        
        param = F.normalize(v - u * (torch.norm(v,2,1,True)/torch.norm(u,2,1,True)),p=2,dim=1)
        torch.save(param.cpu(),parampath)
    return param
    
def localization(data_loader, networks, args, act_threshold=np.arange(0.01,1,0.01), saveimgs=False, topk = None):
    loc = networks[0]
    cls = networks[1]
    [net.eval() for net in networks]
    # init data_loader
    means = [0.485, .456, .406]
    stds = [.229, .224, .225]
    means = torch.reshape(torch.tensor(means), (1, 3, 1, 1)).cuda(args.gpu)
    stds = torch.reshape(torch.tensor(stds), (1, 3, 1, 1)).cuda(args.gpu)

    with torch.no_grad():
        param     = load_param(args,loc,cls)
        locaccacc = tqdm.tqdm(bar_format='{desc}{postfix}')
        maxboxacc = tqdm.tqdm(bar_format='{desc}{postfix}')
        seen      = np.array([0. for _ in range(len(act_threshold))])
        hit_iou70 = np.array([0. for _ in range(len(act_threshold))])
        hit_iou50 = np.array([0. for _ in range(len(act_threshold))])
        hit_iou30 = np.array([0. for _ in range(len(act_threshold))])
        hit_top1  = np.array([0. for _ in range(len(act_threshold))])
        hit_top5  = np.array([0. for _ in range(len(act_threshold))])
        iou70     = np.array([0. for _ in range(len(act_threshold))])
        iou50     = np.array([0. for _ in range(len(act_threshold))])
        iou30     = np.array([0. for _ in range(len(act_threshold))])
        top1      = np.array([0. for _ in range(len(act_threshold))])
        top5      = np.array([0. for _ in range(len(act_threshold))])
        
        for data in tqdm.tqdm(data_loader):
            if args.cls_network == 'N':
                img, _, img_id, target, bbox = data
            else:
                img, _, img_id, target, bbox, cls_img  = data
            
            img = img.cuda(args.gpu, non_blocking=True)

            feat = loc(img)
            
            N,C,H,W=feat.shape
            feat=feat.reshape(N,C,-1)
            
            target,topk = target,None
            classes=torch.zeros(img.shape[0],dtype=torch.int64)
            
            if args.cls_network != 'N':
                cls_img = cls_img.cuda(args.gpu, non_blocking=True)
                logit = cls(cls_img)
                topk = (logit >= logit[torch.arange(0,img.shape[0],1), target].unsqueeze(1)).sum(1)
                
            if args.classwise:
                classes=target.clone()

            if args.caam:
                attmap = torch.sum(feat,1,True).reshape(N,1,H,W)
            else:
                attmap = torch.sum((F.normalize(feat,p=2,dim=1)*param[classes].unsqueeze(-1)),1,True).reshape(N,1,H,W)

            img_ = img * stds + means
            img_ = img_.cpu().detach().numpy()

            attmap = norm_att_map(attmap.clamp(0))

            attmap = F.interpolate(attmap, (img.size(2), img.size(3)), mode='bilinear', align_corners=True)
            attmap = attmap.cpu().detach().numpy()
            
            img_ = np.transpose(img_, (0, 2, 3, 1))
            attmap = np.transpose(attmap, (0, 2, 3, 1))

            s, s70, s50, s30, t1, t5 = evaluation(img_, img_id, attmap, bbox, args, topk, act_threshold, args.res_dir if saveimgs else None)
            
            seen+=s
            hit_iou70+=s70
            hit_iou50+=s50
            hit_iou30+=s30
            if topk is not None:
                hit_top1+=t1
                hit_top5+=t5
                
            iou70 = (hit_iou70 / seen) * 100
            iou50 = (hit_iou50 / seen) * 100
            iou30 = (hit_iou30 / seen) * 100
            top1  = (hit_top1  / seen) * 100
            top5  = (hit_top5  / seen) * 100
            maxboxaccv2 = (iou70 + iou50 + iou30) / 3.
            
            locaccacc.set_description(('BestThreshold: [{bestth:.2f}]'.format(bestth=act_threshold[maxboxaccv2.argmax()],) if len(act_threshold) > 1 else "") +\
                                       'Loc Acc: IOU30: [{cor3:.3f}] IOU50: [{cor5:.3f}] IOU70: [{cor7:.3f}]'.format(
                                        cor3=iou30.max(), cor5=iou50.max(), cor7=iou70.max()))
            bestt=np.argmax(iou50)
            maxboxacc.set_description('MaxBoxAccV2: [{mbv2:.3f}]'.format(
                                        mbv2=maxboxaccv2.max()) +
                                        (' Top-1 Loc Acc: [{t1: .3f}] Top-5 Loc Acc: [{t5: .3f}]'.format(t1=top1[bestt], t5=top5[bestt])))

    return {"MaxBoxAccV2": float(maxboxaccv2.max()),'IOU30': float(iou30.max()), 'IOU50': float(iou50.max()), 'IOU70': float(iou70.max()),
            "Top-1": float(top1.max()), "Top-5": float(top5.max()), "BestThreshold": float(act_threshold[maxboxaccv2.argmax()])}

def segmentation(data_loader, networks, args, saveimgs=False):
    # set nets
    loc = networks[0]
    cls = networks[1]
    [net.eval() for net in networks]
    # init data_loader
    val_iter = iter(data_loader)

    means = [.485, .456, .406]
    stds = [.229, .224, .225]
    means = torch.reshape(torch.tensor(means), (1, 3, 1, 1)).cuda(args.gpu)
    stds = torch.reshape(torch.tensor(stds), (1, 3, 1, 1)).cuda(args.gpu)

    with torch.no_grad():
        param = load_param(args,loc,cls)
        mioudesc = tqdm.tqdm(bar_format='{desc}{postfix}')
        threshold_list_right_edge=np.arange(0,1,0.01)
        threshold_list_right_edge=np.append(threshold_list_right_edge,[1.0, 2.0, 3.0])
        num_bins=threshold_list_right_edge.shape[0]-1
        gt_true_score_hist = np.zeros(num_bins, dtype=np.float)
        gt_false_score_hist = np.zeros(num_bins, dtype=np.float)
        
        for data in tqdm.tqdm(data_loader):
            img, _, img_id, target, gt_mask = data
            img = img.cuda(args.gpu, non_blocking=True)
            
            feat = loc(img)
            
            N,C,H,W=feat.shape
            feat=feat.reshape(N,C,-1)
               
            if W.shape[0]==1:
                classes=torch.zeros(img.shape[0],dtype=torch.int64)
            else:
                classes=target.clone()

            if args.caam:
                attmap = torch.sum(feat,1,True).reshape(N,1,H,W)
            else:
                attmap = torch.sum((F.normalize(feat,p=2,dim=1)*param[classes].unsqueeze(-1)),1,True).clamp(0).reshape(N,1,H,W)
            
            img_ = img * stds + means
            img_ = img_.cpu().detach().numpy()
            gt_mask = gt_mask.cpu().detach().numpy()
            gt_mask = np.round(gt_mask).astype(np.uint8)
            attmap = norm_att_map(attmap)
            attmap = F.interpolate(attmap, (img.size(2), img.size(3)), mode='bilinear',align_corners=True)
            attmap = attmap.cpu().detach().numpy()
            img_ = np.transpose(img_, (0, 2, 3, 1))
            attmap = np.transpose(attmap, (0, 2, 3, 1))

            gt_true_scores = attmap[gt_mask == 1]
            gt_false_scores = attmap[gt_mask == 0]
            # histograms in ascending order
            gt_true_hist, _ = np.histogram(gt_true_scores,
                                        bins=threshold_list_right_edge)
            gt_true_score_hist += gt_true_hist.astype(np.float)

            gt_false_hist, _ = np.histogram(gt_false_scores,
                                            bins=threshold_list_right_edge)
            gt_false_score_hist += gt_false_hist.astype(np.float)

            num_gt_true = gt_true_score_hist.sum()
            tp = gt_true_score_hist[::-1].cumsum()
            fn = num_gt_true - tp

            num_gt_false = gt_false_score_hist.sum()
            fp = gt_false_score_hist[::-1].cumsum()
            tn = num_gt_false - fp

            if ((tp + fn) <= 0).all():
                raise RuntimeError("No positive ground truth in the eval set.")
            if ((tp + fp) <= 0).all():
                raise RuntimeError("No positive prediction in the eval set.")

            non_zero_indices = (tp + fp) != 0
            iou = tp / (num_gt_true + fp)*100.
            
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)

            auc = (precision[1:] * np.diff(recall))[non_zero_indices[1:]].sum()
            auc *= 100
            
            mioudesc.set_description('auc: [{a:.3f}], piou: [{i:.3f}]'.format(a = auc,i=iou.max()))
            
            if saveimgs:
                for bidx in range(img.shape[0]):
                    _,cammed = cammed_image(img_[bidx], attmap[bidx])
                    cv2.imwrite(os.path.join(args.res_dir,f"{img_id[bidx]}.jpg"),cv2.cvtColor(cammed.astype(np.uint8),cv2.COLOR_RGB2BGR))
    return {'PxAP':auc, 'p-IOU':iou.max()}

def evaluation(img,img_id,attmap,bbox,args,topk,act_threshold,savename=None):
    seen      = np.array([0 for _ in range(len(act_threshold))])
    hit_iou70 = np.array([0 for _ in range(len(act_threshold))])
    hit_iou50 = np.array([0 for _ in range(len(act_threshold))])
    hit_iou30 = np.array([0 for _ in range(len(act_threshold))])
    t1        = np.array([0 for _ in range(len(act_threshold))])
    t5        = np.array([0 for _ in range(len(act_threshold))])
    
    for i,t in enumerate(act_threshold):
        for bidx in range(img.shape[0]):
            gray_heatmap = np.squeeze(attmap[bidx]*255).astype(np.uint8)
            th_val = t * np.max(gray_heatmap)
            _, th_gray_heatmap = cv2.threshold(gray_heatmap, int(th_val), 255, cv2.THRESH_BINARY)
            
            try:
                _, contours, _ = cv2.findContours(th_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            except:
                contours, _ = cv2.findContours(th_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                
            bbox_ = np.frombuffer(bbox[bidx],np.float32).reshape(-1,4)
            seen[i]+=1
            if len(contours) != 0:
                #def keyfunc(a):
                #    mask = np.zeros_like(gray_heatmap)
                #    mask = cv2.drawContours(mask,[a],0,1,-1)
                #    return (gray_heatmap.astype(np.float32)*mask).astype(np.float32).sum()
                #c = max(contours, key=keyfunc)
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                estimated_box = [x, y, x + w, y + h]
                IOU = calculate_IOU(bbox_, estimated_box)
                
                if IOU>=0.7:
                    hit_iou70[i]+=1
                if IOU >= 0.5:
                    hit_iou50[i] += 1
                    if topk is not None:
                        if topk[bidx] <= 1:
                            t1[i]+=1
                        if topk[bidx] <= 5:
                            t5[i]+=1
                if IOU >= 0.3:
                    hit_iou30[i] += 1
                

    if not savename is None:
        for bidx in range(img.shape[0]):
            gray_heatmap = np.squeeze(attmap[bidx]*255).astype(np.uint8)
            th_val = act_threshold[hit_iou50.argmax()] * np.max(gray_heatmap)
            _, th_gray_heatmap = cv2.threshold(gray_heatmap, int(th_val), 255, cv2.THRESH_BINARY)

            try:
                _, contours, _ = cv2.findContours(th_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            except:
                contours, _ = cv2.findContours(th_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            bbox_ = np.frombuffer(bbox[bidx],np.float32).reshape(-1,4)
            if len(contours) != 0:
                #def keyfunc(a):
                #    mask = np.zeros_like(gray_heatmap)
                #    mask = cv2.drawContours(mask,[a],0,1,-1)
                #    return (gray_heatmap.astype(np.float32)*mask).astype(np.float32).sum()
                #c = max(contours, key=keyfunc)
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                estimated_box = [x, y, x + w, y + h]
                IOU = calculate_IOU(bbox_, estimated_box)
                
                _, cammed = cammed_image(img[bidx], attmap[bidx])
                #cv2.imwrite(os.path.join(savename,f"{img_id[bidx]}c.jpg"),cv2.cvtColor(cammed.astype(np.uint8),cv2.COLOR_RGB2BGR))
                #for bb in bbox_:
                #    gxa = int(bb[0])
                #    gya = int(bb[1])
                #    gxb = int(bb[2])
                #    gyb = int(bb[3])
                #    cammed = cv2.rectangle(cammed, (max(1, gxa), max(1, gya)),
                #                        (min(args.image_size + 1, gxb), min(args.image_size + 1, gyb)), (255, 0, 0),
                #                        2)
                    
                cammed = cv2.rectangle(cammed, (max(1, estimated_box[0]), max(1, estimated_box[1])),
                                    (min(args.image_size + 1, estimated_box[2]),
                                        min(args.image_size + 1, estimated_box[3])), (0,255,0), 2)
                
                cv2.imwrite(os.path.join(savename,f"{img_id[bidx]}.jpg"),cv2.cvtColor(cammed.astype(np.uint8),cv2.COLOR_RGB2BGR))
                binary = np.zeros_like(cammed.astype(np.uint8))
                binary = cv2.drawContours(binary, [c], 0, color = (255,255,255), thickness=-1)
                cv2.imwrite(os.path.join(savename,f"{img_id[bidx]}b.jpg"),binary)
                cv2.imwrite(os.path.join(savename,f"{img_id[bidx]}o.jpg"),cv2.cvtColor((img[bidx]*255).astype(np.uint8),cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(os.path.join(savename,f"{img_id[bidx]}.jpg"),cv2.cvtColor((img[bidx]*255).astype(np.uint8),cv2.COLOR_RGB2BGR))

    return seen, hit_iou70, hit_iou50, hit_iou30, t1, t5
