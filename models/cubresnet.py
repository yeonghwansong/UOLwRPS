import torch.nn as nn
from torchvision import models
from torch.nn import init

def weight_init_kaiming(m):
    class_names = m.__class__.__name__
    if class_names.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif class_names.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
        
class ResNet(nn.Module):
    def __init__(self, pre_trained=True, n_class=200, model_choice=50):
        super(ResNet, self).__init__()
        self.n_class = n_class
        self.base_model = self._model_choice(pre_trained, model_choice)
        self.base_model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.base_model.fc = nn.Linear(2048, n_class)
        self.base_model.fc.apply(weight_init_kaiming)

    def forward(self, x):
        N = x.size(0)
        x = self.base_model(x)
        return x

    def _model_choice(self, pre_trained, model_choice):
        if model_choice == 50:
            return models.resnet50(pretrained=pre_trained)
        elif model_choice == 101:
            return models.resnet101(pretrained=pre_trained)
        elif model_choice == 152:
            return models.resnet152(pretrained=pre_trained)