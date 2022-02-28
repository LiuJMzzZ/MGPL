import torch
import torch.nn as nn
import torch.nn.functional as F

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class LateralConv2d(nn.Module):

    def __init__(self, in_channels, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self.lateral_conv = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),  # prelu leakyrelu
        )
    def forward(self, x):
        x = self.lateral_conv(x)
        return x

class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18Enc(nn.Module):
    
    def __init__(self, nc=3, z_dim=10, num_Blocks=[2,2,2,2]):
    # def __init__(self, nc=3, z_dim=10, num_Blocks=[3,4,6,3]):
        super().__init__()
        self.in_planes = 64
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.mu = nn.Linear(512, z_dim)
        self.logvar = nn.Linear(512, z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = F.adaptive_avg_pool2d(x4, 1)
        x = x.view(x.size(0), -1)
        mu = self.mu(x)
        logvar = self.logvar(x)
        mid_x = {
            'x_l1': x1,
            'x_l2': x2,
            'x_l3': x3,
            'x_l4': x4,
        }
        return mu, logvar, mid_x


class ResNet18Dec(nn.Module):

    def __init__(self, nc=3, z_dim=10, num_Blocks=[2,2,2,2]):
        super().__init__()
        self.in_planes = 512
        self.drop_rate = 0.9
        self.linear = nn.Linear(z_dim, 512)
        self.nc = nc

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=1)
        self.conv_l4 = LateralConv2d(512, self.drop_rate)
        self.conv_l3 = LateralConv2d(256, self.drop_rate)
        self.conv_l2 = LateralConv2d(128, self.drop_rate)
        self.conv_l1 = LateralConv2d(64, self.drop_rate)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z, mid_x):
        scale=mid_x['x_l4'].size(2) // 4
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x4 = F.interpolate(x, scale_factor=4*scale) # 4 for input 32*32 image
        # x4 = torch.zeros_like(x4) # 512 * 4 * 4
        x4 = x4 + self.conv_l4(mid_x['x_l4']) # 512 * 4 * 4
        x3 = self.layer4(x4)                  # 256 * 8 * 8
        x3 = x3 + self.conv_l3(mid_x['x_l3']) # 256 * 8 * 8
        x2 = self.layer3(x3)                  # 128 * 16 * 16
        x2 = x2 + self.conv_l2(mid_x['x_l2']) # 128 * 16 * 16
        x1 = self.layer2(x2)                  # 64 * 32 * 32
        x1 = x1 + self.conv_l1(mid_x['x_l1']) # 64 * 32 * 32
        x = self.layer1(x1)                   # 64 * 32 * 32
        x = torch.sigmoid(self.conv1(x1))
        x = x.view(x.size(0), self.nc, 32*scale, 32*scale)
        return x


if __name__ == '__main__':
    net = ResNet18Enc()
    print(net)