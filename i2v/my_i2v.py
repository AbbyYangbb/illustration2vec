import numpy as np
import torch as t
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
# from skimage import io
import os



class My_i2v(t.nn.Module):
    def __init__(self, num_classes=1539, init_weignetts =True):
        super(My_i2v, self).__init__()
        if not num_classes:
            self.num_classes = danbooru_2_illust2vec.DEFAULT_NUM_CLASSES
            
        '''
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)        

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)                
        self.conv6_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6_4 = nn.Conv2d(512, num_classes, kernel_size=3, stride=1, padding=1)

        self.prediction_fn = nn.Sigmoid()
        
        '''    
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),            
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        
        self.classifier = nn.Sequential(
#             nn.Dropout(p=0.5),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)                ,
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, num_classes, kernel_size=3, stride=1, padding=1),
        )              

    def forward(self, X):
        '''    
        net = F.relu(self.conv1_1(X))
        net = F.max_pool2d(net, kernel_size=2, stride=2)

        net = F.relu(self.conv2_1(net))
        net = F.max_pool2d(net, kernel_size=2, stride=2)

        net = F.relu(self.conv3_1(net))
        net = F.relu(self.conv3_2(net))
        net = F.max_pool2d(net, kernel_size=2, stride=2)

        net = F.relu(self.conv4_1(net))
        net = F.relu(self.conv4_2(net))
        net = F.max_pool2d(net, kernel_size=2, stride=2)

        net = F.relu(self.conv5_1(net))
        net = F.relu(self.conv5_2(net))
        net = F.max_pool2d(net, kernel_size=2, stride=2)

        net = F.relu(self.conv6_1(net))
        net = F.relu(self.conv6_2(net))
        net = F.relu(self.conv6_3(net))
        net = self.conv6_4(net) # net = F.relu(self.conv6_4(net))

        net = net.view(net.size(0), -1)  # linearized the output of the module 'features'        

        net = F.avg_pool2d(net, kernel_size=7, stride=2)
        net = self.prediction_fn(net)
        '''    
        out = out = self.features(X)
        out = self.classifier(out)
        out = out.view(out.size(0), -1)
        out = F.sigmoid(out)
        return out

class TrainModel:
    def __init__(self):
        def prep_data():
            pass

        # # Batch Sizes for dataloaders
        # self.train_batch_size = 100  # total 500*200 images, 1000 batches of 100 images each
        # self.validation_batch_size = 10  # total 10000 images, 10 batches of 1000 images each            
        
#         def init_vgg19(model_folder):
#             if not os.path.exists(os.path.join(model_folder, 'vgg19.weight')):
#                 pretrained_vgg11 = models.vgg11(pretrained=True)
#             # Copying weights from pretrained model to my model for all layers except the last linear layer
#             for (src, dst) in zip(pretrained_vgg11.parameters(), self.model.parameters()):
#                 dst.data[:] = src.data
#             t.save(self.model.state_dict(), os.path.join(model_folder, 'vgg19.weight'))
        
        t.manual_seed(1)    
        self.model = My_i2v()
#         init_vgg19('./models')
#         vgg.load_state_dict(t.load('./models/vgg19.weight')) 

        # Freeze the weights of all the layers of the new model (with pre-trained weights from pre-trained model) except
        # the last linear layer
        for param in self.model.parameters():
            param.requires_grad = False
        # We need to set enable gradient calculation for the last layer, as it has been set to False earlier
        '''
        for param in self.model.conv6_1.parameters():
            param.requires_grad = True
        for param in self.model.conv6_2.parameters():
            param.requires_grad = True
        for param in self.model.conv6_3.parameters():
            param.requires_grad = True
        for param in self.model.conv6_4.parameters():
            param.requires_grad = True
        '''

        for param in self.model.classifier[0].parameters():
            param.requires_grad = True
        for param in self.model.classifier[2].parameters():
            param.requires_grad = True
        for param in self.model.classifier[4].parameters():
            param.requires_grad = True
        for param in self.model.classifier[5].parameters():
            param.requires_grad = True

        if t.cuda.is_available():
            self.model.cuda()

        # Hyper-parameters for Training
        self.learning_rate = 0.001  # learning rate for optimizer
        self.epochs = 50  # no of times training and validation to be performed on network

        # Set loss function as Cross Entropy Loss
        self.loss_fn = nn.CrossEntropyLoss()  # according to the main paper (i2v)

        opt_params = list(self.model.classifier[0].parameters()) \
        + list(self.model.classifier[2].parameters()) \
        + list(self.model.classifier[4].parameters()) \
        + list(self.model.classifier[5].parameters())

        self.optimizer = t.optim.Adam(opt_params, lr=self.learning_rate)

#         if t.cuda.is_available():
#             Variable(opt_params.data).cuda()

        print(self.model)
        
test = TrainModel()
