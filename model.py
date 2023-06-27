import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class CNN_Net(nn.Module):
    def __init__(self, model, feature_size, num_ftrs, classes_num):
        super(CNN_Net, self).__init__()

        self.backbone = model
        self.num_ftrs = num_ftrs
        self.img_size = 448
        # self.part_size = 224

        # stage 1
        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs//4, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True),
            nn.AdaptiveMaxPool2d(1)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )

        # stage 2
        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs//2, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True),
            nn.AdaptiveMaxPool2d(1)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )

        # stage 3
        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True),
            nn.AdaptiveMaxPool2d(1)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )
        
        # concat features from different stages
        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2 * 3),
            nn.Linear(self.num_ftrs//2 * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

    def forward(self, x):
        _, _, f1, f2, f3 = self.backbone(x)

        batch = x.shape[0]

        f1 = self.conv_block1(f1).view(batch, -1)
        f2 = self.conv_block2(f2).view(batch, -1)
        f3 = self.conv_block3(f3).view(batch, -1)
        y1 = self.classifier1(f1)
        y2 = self.classifier2(f2)
        y3 = self.classifier3(f3)
        y4 = self.classifier_concat(torch.cat((f1, f2, f3), -1))

        return y1, y2, y3, y4, f1, f2, f3
    
    def get_params(self):
        finetune_layer_params = list(self.backbone.parameters())
        finetune_layer_params_ids = list(map(id, finetune_layer_params))
        fresh_layer_params = filter(lambda p: id(p) not in finetune_layer_params_ids, self.parameters())
        return finetune_layer_params, fresh_layer_params
    
    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
