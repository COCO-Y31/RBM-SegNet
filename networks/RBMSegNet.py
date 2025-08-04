import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('./')
from networks.BAM import *


class ResidualBlockGS(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlockGS, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class UDFFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_scale=2, a1=0.7, a2=0.3):
        super(UDFFBlock, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.upconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=upsample_scale, stride=upsample_scale)

        self.a1 = a1
        self.a2 = a2

    def forward(self, input_A, input_B, input_C):
        F_map = self.conv1x1(input_A) + input_B
        M = self.a1 * F_map + self.a2 * input_C
        output = self.upconv(M)
        return M, output


class RBMSegNet(nn.Module):
    def __init__(self,input_nbr,label_nbr):
        super(RBMSegNet, self).__init__()

        batchNorm_momentum = 0.1

        self.bam1 = BAM(64)
        self.bam2 = BAM(128)
        self.bam3 = BAM(256)
        self.bam4 = BAM(512)
        self.bam5 = BAM(512)

        self.conv0_0 = nn.Conv2d(input_nbr, 64, kernel_size=3, padding=1)
        self.bn0_0 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        # self.bn0_0 = nn.GroupNorm(8, 64)  # 替换 BatchNorm2d
        self.conv0_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn0_1 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        # self.bn0_1 = nn.GroupNorm(8, 64)  # 替换 BatchNorm2d

        self.conv1_0 = ResidualBlockGS(64, 128, stride=1)

        self.conv2_0 = ResidualBlockGS(128, 256, stride=1)

        self.conv3_0 = ResidualBlockGS(256, 512, stride=1)

        self.conv4_0 = ResidualBlockGS(512, 512, stride=1)

        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        # self.bn52d = nn.GroupNorm(8, 512)  # 替换 BatchNorm2d
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        # self.bn51d = nn.GroupNorm(8, 512)  # 替换 BatchNorm2d

        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        # self.bn42d = nn.GroupNorm(8, 512)  # 替换 BatchNorm2d
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        # self.bn41d = nn.GroupNorm(8, 256)  # 替换 BatchNorm2d

        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        # self.bn32d = nn.GroupNorm(8, 256)  # 替换 BatchNorm2d
        self.conv31d = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        # self.bn31d = nn.GroupNorm(8, 128)  # 替换 BatchNorm2d

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        # self.bn22d = nn.GroupNorm(8, 128)  # 替换 BatchNorm2d
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        # self.bn21d = nn.GroupNorm(8, 64)  # 替换 BatchNorm2d

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        # self.bn12d = nn.GroupNorm(8, 64)  # 替换 BatchNorm2d
        self.conv11d = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)

        self.UDFF41 = UDFFBlock(256, 256)
        self.UDFF31 = UDFFBlock(128, 128)
        self.UDFF21 = UDFFBlock(64, 64)

        self.conv1x14 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1x13 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1x12 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        # self.bn41 = nn.GroupNorm(8, 512)  # 替换 BatchNorm2d
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        # self.bn42 = nn.GroupNorm(8, 512)  # 替换 BatchNorm2d



    def forward(self, x):

        # Stage 1
        # 256 256 5
        x11 = F.relu(self.bn0_0(self.conv0_0(x)))
        # 256 256 64
        x12 = F.relu(self.bn0_1(self.conv0_1(x11)))
        # 256 256 64
        x12 = self.bam1(x12)
        x1p, id1 = F.max_pool2d(x12,kernel_size=2, stride=2,return_indices=True)
        # 128 128 64

        # Stage 2
        x21 = self.conv1_0(x1p)
        # 128 128 128
        x22 = self.bam2(x21)
        x2p, id2 = F.max_pool2d(x22,kernel_size=2, stride=2,return_indices=True)
        # 64 64 128

        # Stage 3
        x31 = self.conv2_0(x2p)
        # 64 64 256
        x32 = self.bam3(x31)
        # x33 = F.relu(self.bn33(self.conv33(x32)))
        x3p, id3 = F.max_pool2d(x32,kernel_size=2, stride=2,return_indices=True)
        # 32 32 256

        # Stage 4
        x41 = self.conv3_0(x3p)
        # 32 32 512
        x42 = self.bam4(x41)
        # x43 = F.relu(self.bn43(self.conv43(x42)))
        x4p, id4 = F.max_pool2d(x42,kernel_size=2, stride=2,return_indices=True)
        # 16 16 512

        # Stage 5
        x51 = self.conv4_0(x4p)
        # 16 16 512
        x52 = self.bam5(x51)
        # x53 = F.relu(self.bn53(self.conv53(x52)))
        x5p, id5 = F.max_pool2d(x52,kernel_size=2, stride=2,return_indices=True)
        # 8 8 512

        # Stage 5d
        x5p = self.bam5(x5p)
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
        # x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x5d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        x51d = self.bam4(x51d)
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
        # x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x4d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))
        x4_1 = F.relu(self.bn41(self.conv41(x3p)))
        # 32 32 512
        x4_2 = F.relu(self.bn42(self.conv42(x4_1)))
        x41f, x41ff = self.UDFF41(x3p, x41d, self.conv1x14(x4_2))
        print(x3p.shape,x41d.shape,x4_2.shape,x41f.shape,x41ff.shape)

        # Stage 3d
        x41d = self.bam3(x41f)
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        # x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x3d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))
        x31f, x31ff = self.UDFF31(x2p, x31d, self.conv1x13(x41ff))

        # Stage 2d
        x31d = self.bam2(x31f)
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))
        x21f, x21ff = self.UDFF21(x1p, x21d, self.conv1x12(x31ff))

        # Stage 1d
        x21d = self.bam1(x21f)
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)

        return x11d


if __name__=="__main__":
    model = RBMSegNet(4,2)
    print(model)
    a = torch.randn(2,4,256,256)
    out = model(a)
    print(type(out))
