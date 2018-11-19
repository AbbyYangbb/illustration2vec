import numpy as np
import torch as t

# import torch.nn as nn
import import torchvision.models as models
import torchvision.transform as T
import torch.nn.functional as F

from torch.autograd import Variable
import torch.optim as optim

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights =True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), 
            nn.ReLU(True),
            nn.Dropout(),
            Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(), 
            nn.Linear(4096, num_classes), 
        )
        if init_weights:
            self._initialize_weights()
            
    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear): # FC
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
            # different layer types have different way of initialization
            
    def make_layers(cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if x = 'M': # MaxPool2d
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else: # Conv2d
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                # v? name
                if batch_norm: # add BN bt Conv and ReLU
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.RelU(inplace=True)]
                in_channels =  v
        return nn.Sequential(*layers)
    
    cfg = {
        'A' : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', ]
        #  conv2d, Pool
        # vgg: all 5 blocks
    }
    
    def vgg11(pretrained=False, **kwargs):
        if pretrained:
            kwargs['init_weights'] = Flase
        model = VGG(make_layers(cfg['A']), **kwargs)
        #           conv layers(get features)
        # def __init__(self, features, num_classes=1000, init_weights=True):
        
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
        return model
    
    # modify to i2v
                    
                    
                    
                
                 
            
