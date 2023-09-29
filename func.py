import os
import torch
import numpy as np
from torch.autograd import Variable
import cv2
import matplotlib.pyplot as plt
from torch.nn import functional as F
import xml.etree.ElementTree as ET

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def formatted_print(notice, value):
    print('{0:<40}{1:<40}'.format(notice, value))


def save_checkpoint(state, check_list, log_dir, epoch=0):
    check_file = os.path.join(log_dir, 'model_{}.ckpt'.format(epoch))
    torch.save(state, check_file)
    check_list.write('model_{}.ckpt\n'.format(epoch))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        pred = output.argmax(1)
        correct = pred.eq(target)
        correct = correct.float().sum()*(25./output.shape[0])
        return [correct,]



def mixup_data(x, y, alpha=0.2, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(pred, y_a, y_b, prob):
    return prob * F.cross_entropy(pred, y_a) + (1 - prob) * F.cross_entropy(pred, y_b)


def mix_data(x, x_flip):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    # lam = np.random.uniform(0.0, 1.0)
    lam = np.random.beta(0.4, 0.4)

    x_mix = lam * x + (1 - lam) * x_flip
    return x_mix, lam


def get_att_map(feats, y_, norm=True):
    with torch.no_grad():
        y_ = y_.long()
        att_map = Variable(torch.zeros([feats.shape[0], feats.shape[2], feats.shape[3]]))

        for idx in range(feats.shape[0]):
            att_map[idx, :, :] = torch.squeeze(feats[idx, y_.data[idx], :, :])

        if norm:
            att_map = norm_att_map(att_map)

    return att_map


def norm_att_map(att_maps):
    _min = att_maps.min(-1, keepdim=True)[0].min(-2, keepdim=True)[0]
    _max = att_maps.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
    att_norm = (att_maps - _min) / (_max - _min)
    return att_norm

def batch_norm_att_map(att_maps):
    _min = att_maps.min()
    _max = att_maps.max()
    att_norm = (att_maps - _min) / (_max - _min)
    return att_norm

def batch_norm_att_map_pm(att_maps, _max=None, _min=None):
    pos=att_maps>0
    neg=att_maps<=0
    att_maps=att_maps.clone()
    att_maps[pos]=torch.log(att_maps[pos]*2+1)
    if _max is None:
        att_maps[pos]/=att_maps[pos].max()+1e-10
    else:
        att_maps[pos]/=torch.log(_max*2+1)
    if neg.sum()>0:
        att_maps[neg]=torch.log(-att_maps[neg]*2+1)
        if _min is None:
            att_maps[neg]/=att_maps[neg].max()+1e-10
        else:
            att_maps[neg]/=torch.log(-_min*2+1)
        att_maps[neg]*=-1
    att_maps+=1.
    att_maps/=2.
    return att_maps

#def batch_norm_att_map_pm(att_maps):
#    _max = att_maps.abs().max()
#    att_norm = att_maps / _max
#    att_norm += 1.
#    att_norm /= 2.
#    return att_norm

def load_imagenet_bbox(args, dataset_path='/data2/imagenet/val/'):
    origin_bbox = {}
    image_sizes = {}
    resized_bbox = {}
    #with open(os.path.join(dataset_path, 'val.txt')) as f:
    #    for image_id, each_line in enumerate(f):
    #        if len(each_line)<3:
    #            continue
    #        file_info = each_line.strip().split()
    for fname in os.listdir(dataset_path):
        if not fname.endswith("xml"):
            continue
        tree = ET.ElementTree(file=os.path.join(dataset_path, fname))
        root = tree.getroot()
        ObjectSet = root.findall('object')
        bbox_line = []
        for Object in ObjectSet:
            BndBox = Object.find('bndbox')
            xmin = BndBox.find('xmin').text
            ymin = BndBox.find('ymin').text
            xmax = BndBox.find('xmax').text
            ymax = BndBox.find('ymax').text
            xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
            bbox_line.append([xmin, ymin, xmax, ymax])
        image_id = int(fname.split("_")[-1].split(".")[0])
        origin_bbox[image_id] = bbox_line
        image_sizes[image_id] = [int(root.find('size').find('width').text), int(root.find('size').find('height').text)]

    for i in origin_bbox.keys():
        image_width, image_height = image_sizes[i]
        bbox = []
        for x, y, xmax, ymax in origin_bbox[i]:
            x_scale = args.image_size / image_width
            y_scale = args.image_size / image_height
            x_new = np.clip(int(np.round(x * x_scale) - (args.image_size - args.crop_size) * 0.5), 0, args.crop_size)
            y_new = np.clip(int(np.round(y * y_scale) - (args.image_size - args.crop_size) * 0.5), 0, args.crop_size)
            x_max = np.clip(int(np.round(xmax * x_scale) - (args.image_size - args.crop_size) * 0.5), 0, args.crop_size)
            y_max = np.clip(int(np.round(ymax * y_scale) - (args.image_size - args.crop_size) * 0.5), 0, args.crop_size)

            bbox.append([x_new, y_new, x_max, y_max])
            
        resized_bbox[i] = bbox

    return resized_bbox

def cammed_image(image, mask, require_norm=False, jet=True):
    if require_norm:
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)
    #mask=np.sqrt(mask)
    heatmap = cv2.applyColorMap(np.uint8(mask * 255.), cv2.COLORMAP_JET)# if jet else cmapy.cmap('cool'))
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    cam = (heatmap + np.float32(image)) if jet else (heatmap*2 + np.float32(image))
    cam = cam / np.max(cam)
    return heatmap * 255., cam * 255.


def intensity_to_rgb(intensity, normalize=False):
    """
    Convert a 1-channel matrix of intensities to an RGB image employing a colormap.
    This function requires matplotlib. See `matplotlib colormaps
    <http://matplotlib.org/examples/color/colormaps_reference.html>`_ for a
    list of available colormap.
    Args:
        intensity (np.ndarray): array of intensities such as saliency.
        cmap (str): name of the colormap to use.
        normalize (bool): if True, will normalize the intensity so that it has
            minimum 0 and maximum 1.
    Returns:
        np.ndarray: an RGB float32 image in range [0, 255], a colored heatmap.
    """
    assert intensity.ndim == 2, intensity.shape
    intensity = intensity.astype("float")

    if normalize:
        intensity -= intensity.min()
        intensity /= intensity.max()

    cmap = 'jet'
    cmap = plt.get_cmap(cmap)
    intensity = cmap(intensity)[..., :3]
    return intensity.astype('float32') * 255.0


def large_rect(rect):
    # find largest recteangles
    large_area = 0
    target = 0
    if len(rect) == 1:
        x = rect[0][0]
        y = rect[0][1]
        w = rect[0][2]
        h = rect[0][3]
        return x, y, w, h
    else:
        for i in range(1, len(rect)):
            area = rect[i][2] * rect[i][3]
            if large_area < area:
                large_area = area
                target = i
        x = rect[target][0]
        y = rect[target][1]
        w = rect[target][2]
        h = rect[target][3]
        return x, y, w, h


def caliou(boxesA, boxesB):
    maxiou=0.
    for boxA in boxesA:
        for boxB in boxesB:
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            # compute the area of intersection rectangle
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            iou = interArea / float(boxAArea + boxBArea - interArea)
            if iou>maxiou:
                maxiou=iou
    return maxiou

def calculate_IOU(boxesA, boxB):
    maxiou=0.
    for boxA in boxesA:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        if iou>maxiou:
            maxiou=iou
            
    # return the intersection over union value
    return maxiou


def calculate_IOU_multibox(boxA, boxesB):

    interArea = 0.0
    boxAArea = 0.0
    boxBArea = 0.0

    for boxB in boxesB:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea += max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxBArea += (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea += (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def train_val_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-500])
        val_idxs.extend(idxs[-500:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


class EMA(torch.nn.Module):
    def __init__(self, mu=0.999):
        super(EMA, self).__init__()
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def forward(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.mu) * x + self.mu * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average


def linear_rampup(current, rampup_length=16):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


def cross_entropy(input, target):
    """ Cross entropy for one-hot labels
    """
    return -torch.mean(torch.sum(target * F.log_softmax(input), dim=1))


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, args):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, args.lambda_u * linear_rampup(epoch)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


# my Rectangle = (x1, y1, x2, y2), a bit different from OP's x, y, w, h
def intersection(rectA, rectB): # check if rect A & B intersect
    a, b = rectA, rectB
    startX = max(min(a[0], a[2]), min(b[0], b[2]))
    startY = max(min(a[1], a[3]), min(b[1], b[3]))
    endX = min(max(a[0], a[2]), max(b[0], b[2]))
    endY = min(max(a[1], a[3]), max(b[1], b[3]))
    if startX < endX and startY < endY:
        return True
    else:
        return False


def combineRect(rectA, rectB): # create bounding box for rect A & B
    a, b = rectA, rectB
    startX = min(a[0], b[0])
    startY = min(a[1], b[1])
    endX = max(a[2], b[2])
    endY = max(a[3], b[3])
    return (startX, startY, endX, endY)

def cos_sim(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return torch.clamp(sim, min=0.0005, max=0.9995)


def cos_distance(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return 1 - sim

import torch.nn as nn
#class PCA(nn.Module):
#    def __init__(self, n = 3):
#        super(PCA, self).__init__()
#        self.n = n
#
#    def forward(self, features: torch.Tensor):     # features: NC
#        k = features.size(0)
#        x_mean = (features.sum(0, True) / k)
#        #features = features - x_mean
#
#        reshaped_features = features - x_mean
#
#        cov = torch.matmul(reshaped_features.t(), reshaped_features) / k
#        eigval, eigvec = torch.eig(cov, eigenvectors=True)
#
#        compos = eigvec[:, :self.n]
#
#        projected_map = torch.matmul(reshaped_features,compos)
#        
#        return projected_map
class PCAProjectNet(nn.Module):
    def __init__(self, n = 0, normalize = False):
        super(PCAProjectNet, self).__init__()
        self.n = n
        self.normalize = normalize

    def forward(self, features: torch.Tensor):     # features: NCWH
        k = features.size(0) * features.size(2)
        x_mean = (features.sum(2, True).sum(0, True) / k)
        #features = features - x_mean

        reshaped_features = features.permute(1, 0, 2).contiguous().view(features.size(1), -1)

        cov = torch.matmul(reshaped_features, reshaped_features.t()) / k
        eigval, eigvec = torch.eig(cov, eigenvectors=True)

        first_compo = eigvec[:, self.n]
        if self.normalize:
            reshaped_features /= torch.norm(reshaped_features,2,0,True)+1e-6

        projected_map = torch.matmul(first_compo.unsqueeze(0), reshaped_features).view(1, features.size(0), -1)\
            .view(features.size(0), 1, features.size(2))

        maxv = projected_map.max()
        minv = projected_map.min()

        projected_map *= (maxv + minv) / torch.abs(maxv + minv)
        
        return projected_map


def l2_distance(embedded_fg, embedded_bg):
    N, C = embedded_fg.size()

    # embedded_fg = F.normalize(embedded_fg, dim=1)
    # embedded_bg = F.normalize(embedded_bg, dim=1)

    embedded_fg = embedded_fg.unsqueeze(1).expand(N, N, C)
    embedded_bg = embedded_bg.unsqueeze(0).expand(N, N, C)

    return torch.pow(embedded_fg - embedded_bg, 2).sum(2) / C

import torch.nn as nn
# Minimize Similarity, e.g., push representation of foreground and background apart.
class SimMinLoss(nn.Module):
    def __init__(self, margin=0.15, metric='cos', reduction='mean'):
        super(SimMinLoss, self).__init__()
        self.m = margin
        self.metric = metric
        self.reduction = reduction

    def forward(self, embedded_bg, embedded_fg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            raise NotImplementedError
        elif self.metric == 'cos':
            sim = cos_sim(embedded_bg, embedded_fg)
            loss = -torch.log(1 - sim)
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)


# Maximize Similarity, e.g., pull representation of background and background together.
class SimMaxLoss(nn.Module):
    def __init__(self, metric='cos', alpha=0.25, reduction='mean'):
        super(SimMaxLoss, self).__init__()
        self.metric = metric
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, embedded_bg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            raise NotImplementedError

        elif self.metric == 'cos':
            sim = cos_sim(embedded_bg, embedded_bg)
            loss = -torch.log(sim)
            loss[loss < 0] = 0
            _, indices = sim.sort(descending=True, dim=1)
            _, rank = indices.sort(dim=1)
            rank = rank - 1
            rank_weights = torch.exp(-rank.float() * self.alpha)
            loss = loss * rank_weights
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss
        
        
#class GMM():
#    def __init__(self):
#        '''
#        Define a model with known number of clusters and dimensions.
#        input:
#            - k: Number of Gaussian clusters
#            - dim: Dimension 
#            - init_mu: initial value of mean of clusters (k, dim)
#                       (default) random from uniform[-10, 10]
#            - init_sigma: initial value of covariance matrix of clusters (k, dim, dim)
#                          (default) Identity matrix for each cluster
#            - init_pi: initial value of cluster weights (k,)
#                       (default) equal value to all cluster i.e. 1/k
#            - colors: Color valu for plotting each cluster (k, 3)
#                      (default) random from uniform[0, 1]
#        '''
#        self.k = 2
#        self.dim = 1
#        self.mu = torch.Tensor(k.min(), k.max())
#        self.sigma = torch.Tensor(0.01, 0.01)
#        self.pi = np.ones(self.k)/self.k
#        self.colors = torch.rand(k, 3)
#    
#    def init_em(self, X):
#        '''
#        Initialization for EM algorithm.
#        input:
#            - X: data (batch_size, dim)
#        '''
#        self.data = X
#        self.num_points = X.shape[0]
#        self.z = np.zeros((self.num_points, self.k))
#    
#    def e_step(self):
#        '''
#        E-step of EM algorithm.
#        '''
#        for i in range(self.k):
#            self.z[:, i] = self.pi[i] * multivariate_normal.pdf(self.data, mean=self.mu[i], cov=self.sigma[i])
#        self.z /= self.z.sum(axis=1, keepdims=True)
#    
#    def m_step(self):
#        '''
#        M-step of EM algorithm.
#        '''
#        sum_z = self.z.sum(axis=0)
#        self.pi = sum_z / self.num_points
#        self.mu = np.matmul(self.z.T, self.data)
#        self.mu /= sum_z[:, None]
#        for i in range(self.k):
#            j = np.expand_dims(self.data, axis=1) - self.mu[i]
#            s = np.matmul(j.transpose([0, 2, 1]), j)
#            self.sigma[i] = np.matmul(s.transpose(1, 2, 0), self.z[:, i] )
#            self.sigma[i] /= sum_z[i]
#            
#    def log_likelihood(self, X):
#        '''
#        Compute the log-likelihood of X under current parameters
#        input:
#            - X: Data (batch_size, dim)
#        output:
#            - log-likelihood of X: Sum_n Sum_k log(pi_k * N( X_n | mu_k, sigma_k ))
#        '''
#        ll = []
#        for d in X:
#            tot = 0
#            for i in range(self.k):
#                tot += self.pi[i] * multivariate_normal.pdf(d, mean=self.mu[i], cov=self.sigma[i])
#            ll.append(np.log(tot))
#        return np.sum(ll)
#    
#    def plot_gaussian(self, mean, cov, ax, n_std=3.0, facecolor='none', **kwargs):
#        '''
#        Utility function to plot one Gaussian from mean and covariance.
#        '''
#        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
#        ell_radius_x = np.sqrt(1 + pearson)
#        ell_radius_y = np.sqrt(1 - pearson)
#        ellipse = Ellipse((0, 0),
#            width=ell_radius_x * 2,
#            height=ell_radius_y * 2,
#            facecolor=facecolor,
#            **kwargs)
#        scale_x = np.sqrt(cov[0, 0]) * n_std
#        mean_x = mean[0]
#        scale_y = np.sqrt(cov[1, 1]) * n_std
#        mean_y = mean[1]
#        transf = transforms.Affine2D() \
#            .rotate_deg(45) \
#            .scale(scale_x, scale_y) \
#            .translate(mean_x, mean_y)
#        ellipse.set_transform(transf + ax.transData)
#        return ax.add_patch(ellipse)
#
#    def draw(self, ax, n_std=2.0, facecolor='none', **kwargs):
#        '''
#        Function to draw the Gaussians.
#        Note: Only for two-dimensionl dataset
#        '''
#        if(self.dim != 2):
#            print("Drawing available only for 2D case.")
#            return
#        for i in range(self.k):
#            self.plot_gaussian(self.mu[i], self.sigma[i], ax, n_std=n_std, edgecolor=self.colors[i], **kwargs)