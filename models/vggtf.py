import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as vmodels
from torchvision._internally_replaced_utils import load_state_dict_from_url

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'N', 512, 512, 'N'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'N', 512, 512, 'N'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512, 'N'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'N', 512, 512, 512, 512, 'N'],
    'vggcam16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512, 'N'],
    'vggcam19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'N', 512, 512, 512, 512, 'N'],
}


class VGGTF(nn.Module):
    def __init__(self, features):
        super(VGGTF, self).__init__()
        self.features = features
        self.blocks = nn.ModuleList()

        chinter = 512

        if self.with_cam:
            if 'odd' in tftypes:
                print('APPEND ODD')
                self.odd = nn.Conv2d(chinter, self.num_cls[5]*self.max_dup, 1, 1)
                self.blocks.append(self.odd)
            if 'rotation' in tftypes:
                print('APPEND ROTATION')
                self.rotation = nn.Conv2d(chinter, self.num_cls[0]*self.max_dup, 1, 1)
                self.blocks.append(self.rotation)
            if 'translation' in tftypes:
                print('APPEND TRANSLATION')
                self.translation = nn.Conv2d(chinter, self.num_cls[1]*self.max_dup, 1, 1)
                self.blocks.append(self.translation)
            if 'shear' in tftypes:
                print('APPEND SHEAR')
                self.shear = nn.Conv2d(chinter, self.num_cls[2]*self.max_dup, 1, 1)
                self.blocks.append(self.shear)
            if 'hflip' in tftypes:
                print('APPEND HFLIP')
                self.hflip = nn.Conv2d(chinter, self.num_cls[3]*self.max_dup, 1, 1)
                self.blocks.append(self.hflip)
            if 'scale' in tftypes:
                print('APPEND SCALE')
                self.scale = nn.Conv2d(chinter, self.num_cls[4]*self.max_dup, 1, 1)
                self.blocks.append(self.scale)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            if 'rotation' in tftypes:
                print('APPEND ROTATION')
                self.rotation = nn.Linear(chinter, self.num_cls[0])
                self.blocks.append(self.rotation)
            if 'translation' in tftypes:
                print('APPEND TRANSLATION')
                self.translation = nn.Linear(chinter, self.num_cls[1])
                self.blocks.append(self.translation)
            if 'shear' in tftypes:
                print('APPEND SHEAR')
                self.shear = nn.Linear(chinter, self.num_cls[2])
                self.blocks.append(self.shear)
            if 'hflip' in tftypes:
                print('APPEND HFLIP')
                self.hflip = nn.Linear(chinter, self.num_cls[3])
                self.blocks.append(self.hflip)
        nn.init.kaiming_normal_(self.rotation.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.rotation.bias, 0)

        self.first=True

    def forward(self, x,na=False):
        logits = []
        x = self.features(x)
        
        #N,Ch,W,H=x.shape
        #A=torch.detach(x).reshape(N,Ch,-1)
        #Max=A.argmax(-1,keepdim=True)
        #Rep=torch.gather(A,2,Max)
        #Sim=torch.sum(((A/(torch.norm(A,2,1,True)+1e-10))*(Rep/(torch.norm(Rep,2,1,True)+1e-10))),1,True)
        #x=x*Sim.reshape(N,1,W,H)
        
        cams = [x]
        # x = self.inter(x)
        if self.with_cam:
            chatt = x.mean(1)
            #if na:
            #    noattention=chatt-chatt.min()
            #    noattention=noattention/noattention.max()
            #    x=(noattention<0.95).unsqueeze(1).float()*x
            #x=torch.softmax(x,1)
            for block in self.blocks:
                cam = block(x)
                cams.append(cam)
                logit = self.pool(cam).squeeze(-1).squeeze(-1)
                logit=logit.reshape((logit.shape[0],self.max_dup,-1))
                #if self.first:
                #    print(x.min(),x.mean(),x.max())
                #    print(logit)
                #    self.first=False
                #logit=log_sigsoftmax(logit)
                #logit=torch.sigmoid(logit)
                #logit=torch.softmax(logit,1)
                #logit=torch.log(torch.softmax(logit,1))
                logits.append(logit)
            cams.append(chatt)
            #if x.shape[0]<32:
            #    self.first=True
            return logits, cams

        else:
            flat = x.view(x.size(0), -1)

            for block in self.blocks:
                logit = block(flat)
                logits.append(logit)
            return logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_ch=3, batch_norm=False):
    layers = []
    in_channels = in_ch
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v, momentum=0.001), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vggcamtf(vggconfig='vggcam16bn', tftypes=[], tfnums=1, max_dup=1, pretrained=True, **kwargs):
    use_bn = ('bn' in vggconfig)

    if pretrained:
        print('USE PRETRAINED BACKBONE')
        #model = VGGTF(make_layers(cfg[vggconfig[:8]], batch_norm=use_bn),
        #              tftypes=tftypes, tfnums=tfnums, with_cam=True,max_dup=max_dup)
        model = nn.Module()
        model.features = make_layers(cfg[vggconfig[:8]], batch_norm=use_bn)
        if vggconfig == 'vggcam16':
            vgg = vmodels.vgg16(pretrained=True)
        elif vggconfig == 'vggcam19':
            vgg = vmodels.vgg19(pretrained=True)
        elif vggconfig == 'vggcam16bn':
            vgg = vmodels.vgg16_bn(pretrained=True)
        elif vggconfig == 'vggcam19bn':
            vgg = vmodels.vgg19_bn(pretrained=True)
        else:
            print("NOT IMPLEMENTED FOR ", vggconfig)
            exit(-3)
        #pretrained_dict = load_state_dict_from_url("https://data.lip6.fr/cadene/pretrainedmodels/vgg16_bn-6c64b313.pth")
        #pretrained_dict=torch.load("/home/syh/cl/p40/vggwobird.pth")
        print(model.load_state_dict(vgg.state_dict(),False))
        model = model.features
    else:
        model = VGGTF(features=make_layers(cfg[vggconfig[:8]], batch_norm=use_bn),
                      tftypes=tftypes, tfnums=tfnums, with_cam=True,max_dup=max_dup)
    class VGGStep(nn.Module):
        def __init__(self,features) -> None:
            super().__init__()
            self.features = features
        def forward(self, x, ms = False):
            m = []
            for l in self.features:
                x = l(x)
                if isinstance(l, nn.MaxPool2d):
                    m.append(x)
            if ms:
                return m
            return x
    return VGGStep(model)
