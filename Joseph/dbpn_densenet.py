import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import torch.optim as optim
from base_networks import *
from torchvision.transforms import *


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class DBPNDenseNet3(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12, reduction=0.5, bottleneck=True, dropRate=0.0, num_channels =3, base_filter=64, feat=256, num_stages=7, scale_factor=8):
        super(DBPNDenseNet3, self).__init__()
        
        #start of DBPN
        
        if scale_factor == 2:
        	kernel_sr = 6
        	stride_sr = 2
        	padding = 2
        elif scale_factor == 4:
        	kernel_sr = 8
        	stride_sr = 4
        	padding_sr = 2
        elif scale_factor == 8:
        	kernel_sr = 12
        	stride_sr = 8
        	padding_sr = 2
        
        #Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu', norm=None)
        #Back-projection stages
        self.up1 = UpBlock(base_filter, kernel_sr, stride_sr, padding_sr)
        self.down1 = DownBlock(base_filter, kernel_sr, stride_sr, padding_sr)
        self.up2 = UpBlock(base_filter, kernel_sr, stride_sr, padding_sr)
        self.down2 = D_DownBlock(base_filter, kernel_sr, stride_sr, padding_sr, 2)
        self.up3 = D_UpBlock(base_filter, kernel_sr, stride_sr, padding_sr, 2)
        self.down3 = D_DownBlock(base_filter, kernel_sr, stride_sr, padding_sr, 3)
        self.up4 = D_UpBlock(base_filter, kernel_sr, stride_sr, padding_sr, 3)
        self.down4 = D_DownBlock(base_filter, kernel_sr, stride_sr, padding_sr, 4)
        self.up5 = D_UpBlock(base_filter, kernel_sr, stride_sr, padding_sr, 4)
        self.down5 = D_DownBlock(base_filter, kernel_sr, stride_sr, padding_sr, 5)
        self.up6 = D_UpBlock(base_filter, kernel_sr, stride_sr, padding_sr, 5)
        self.down6 = D_DownBlock(base_filter, kernel_sr, stride_sr, padding_sr, 6)
        self.up7 = D_UpBlock(base_filter, kernel_sr, stride_sr, padding_sr, 6)
        #Reconstruction
        self.output_conv = ConvBlock(num_stages*base_filter, num_channels, 3, 1, 1, activation=None, norm=None)
        
        
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    torch.nn.init.kaiming_normal(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    torch.nn.init.kaiming_normal(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
                    
        #start of DenseNet
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        
    def forward(self, x):
        x = self.feat0(x)
        x = self.feat1(x)
        
        h1 = self.up1(x)
        l1 = self.down1(h1)
        h2 = self.up2(l1)
        
        concat_h = torch.cat((h2, h1),1)
        l = self.down2(concat_h)
        
        concat_l = torch.cat((l, l1),1)
        h = self.up3(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        l = self.down3(concat_h)
        
        concat_l = torch.cat((l, concat_l),1)
        h = self.up4(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        l = self.down4(concat_h)
        
        concat_l = torch.cat((l, concat_l),1)
        h = self.up5(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        l = self.down5(concat_h)
        
        concat_l = torch.cat((l, concat_l),1)
        h = self.up6(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        l = self.down6(concat_h)
        
        concat_l = torch.cat((l, concat_l),1)
        h = self.up7(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        x = self.output_conv(concat_h)
        
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)
