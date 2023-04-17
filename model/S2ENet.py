import torch
from torch import nn
from torch.nn import functional as F

class Spatial_Enhance_Module(nn.Module):
    def __init__(self, in_channels, inter_channels=None, size=None):
        """Implementation of SAEM: Spatial Enhancement Module
        args:
            in_channels: original channel size
            inter_channels: channel size inside the block if not specifed reduced to half
        """
        super(Spatial_Enhance_Module, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # dimension == 2
        conv_nd = nn.Conv2d
        # max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.W_z = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
            bn(self.in_channels)
        )

        # define Transformation 1 and 2
        self.T1 = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1),
            bn(self.inter_channels),
            nn.Sigmoid()
        )
        self.T2 = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1),
            bn(self.inter_channels),
            nn.Sigmoid()
        )

        self.dim_reduce = nn.Sequential(
            nn.Conv1d(
                in_channels=size * size,
                out_channels=1,
                kernel_size=1,
                bias=False,
            ),
        )

    def forward(self, x1, x2):
        """
        args
            x: (N, C, H, W)
        """

        batch_size = x1.size(0)

        t1 = self.T1(x1).view(batch_size, self.inter_channels, -1)
        t2 = self.T2(x2).view(batch_size, self.inter_channels, -1)
        t1 = t1.permute(0, 2, 1)
        Affinity_M = torch.matmul(t1, t2)

        Affinity_M = Affinity_M.permute(0, 2, 1)  # B*HW*TF --> B*TF*HW
        Affinity_M = self.dim_reduce(Affinity_M)  # B*1*HW
        Affinity_M = Affinity_M.view(batch_size, 1, x1.size(2), x1.size(3))   # B*1*H*W

        x1 = x1 * Affinity_M.expand_as(x1)

        return x1


class Spectral_Enhance_Module(nn.Module):
    def __init__(self, in_channels, in_channels2, inter_channels=None, inter_channels2=None):
        """Implementation of SEEM: Spectral Enhancement Module
        args:
            in_channels: original channel size
            inter_channels: channel size inside the block
        """
        super(Spectral_Enhance_Module, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.in_channels2 = in_channels2
        self.inter_channels2 = inter_channels2

        if self.inter_channels is None:
            self.inter_channels = in_channels
            if self.inter_channels == 0:
                self.inter_channels = 1
        if self.inter_channels2 is None:
            self.inter_channels2 = in_channels2
            if self.inter_channels2 == 0:
                self.inter_channels2 = 1

        # dimension == 2
        conv_nd = nn.Conv2d
        # max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.W_z = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
            bn(self.in_channels)
        )

        # define Transformation 1 and 2
        self.T1 = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            bn(self.inter_channels),
            nn.Sigmoid()
        )
        self.T2 = nn.Sequential(
            conv_nd(in_channels=self.in_channels2, out_channels=self.inter_channels2, kernel_size=1),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            bn(self.inter_channels2),
            nn.Sigmoid()
        )

        self.dim_reduce = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channels2,
                out_channels=1,
                kernel_size=1,
                bias=False,
            )
        )

    def forward(self, x1, x2):
        """
        args
            x: (N, C, H, W)
        """

        batch_size = x1.size(0)

        t1 = self.T1(x1).view(batch_size, self.inter_channels, -1)
        t2 = self.T2(x2).view(batch_size, self.inter_channels2, -1)
        t2 = t2.permute(0, 2, 1)
        Affinity_M = torch.matmul(t1, t2)

        Affinity_M = Affinity_M.permute(0, 2, 1)  # B*C1*C2 --> B*C2*C1
        Affinity_M = self.dim_reduce(Affinity_M)  # B*1*C1
        Affinity_M = Affinity_M.view(batch_size, x1.size(1), 1, 1)  # B*C1*1*1

        x1 = x1 * Affinity_M.expand_as(x1)

        return x1

class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)

        return out

class SEblock(nn.Module):  # 注意力机制模块
    def __init__(self, channel, r=0.5):  # channel为输入的维度, r为全连接层缩放比例->控制中间层个数
        super(SEblock, self).__init__()
        # 全局均值池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel * r)),  # int(channel * r)取整数
            nn.ReLU(),
            nn.Linear(int(channel * r), channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 对x进行分支计算权重, 进行全局均值池化
        branch = self.global_avg_pool(x)
        branch = branch.view(branch.size(0), -1)

        # 全连接层得到权重
        weight = self.fc(branch)

        # 将维度为b, c的weight, reshape成b, c, 1, 1 与 输入x 相乘
        h, w = weight.shape
        weight = torch.reshape(weight, (h, w, 1, 1))

        # 乘积获得结果
        scale = weight * x
        print("scale:", scale.shape)
        return scale

class AFF(nn.Module):
    # 多特征融合 AFF

    def __init__(self, channels=32, r=2):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


class S2ENet(nn.Module):

    def __init__(self, input_channels, input_channels2, n_classes, patch_size):
        super(S2ENet, self).__init__()

        self.activation = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.planes_a = [128, 64, 32]
        self.planes_b = [8, 16, 32]

        # For image a (7×7×input_channels) --> (7×7×planes_a[0])
        self.conv1_a = conv_bn_relu(input_channels, self.planes_a[0], kernel_size=3, padding=1, bias=True)
        # For image b (7×7×input_channels2) --> (7×7×planes_b[0])
        self.conv1_b = conv_bn_relu(input_channels2, self.planes_b[0], kernel_size=3, padding=1, bias=True)

        # For image a (7×7×planes_a[0]) --> (7×7×planes_a[1])
        self.conv2_a = conv_bn_relu(self.planes_a[0], self.planes_a[1], kernel_size=3, padding=1, bias=True)
        # For image b (7×7×planes_b[0]) --> (7×7×planes_b[1])
        self.conv2_b = conv_bn_relu(self.planes_b[0], self.planes_b[1], kernel_size=3, padding=1, bias=True)

        # For image a (7×7×planes_a[1]) --> (7×7×planes_a[2])
        self.conv3_a = conv_bn_relu(self.planes_a[1], self.planes_a[2], kernel_size=3, padding=1, bias=True)
        # For image b (7×7×planes_b[1]) --> (7×7×planes_b[2])
        self.conv3_b = conv_bn_relu(self.planes_b[1], self.planes_b[2], kernel_size=3, padding=1, bias=True)

        self.branch2 = nn.Sequential(
            nn.MaxPool2d(3, 1, padding=1, dilation=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, 1, padding=1)
        )

        self.branch_SE = SEblock(channel=64)

        self.branch_AFF = AFF()

        self.w_hsl = nn.Parameter(torch.ones(4))
        self.w_lidar = nn.Parameter(torch.ones(4))

        self.SAEM = Spatial_Enhance_Module(in_channels=self.planes_a[2], inter_channels=self.planes_a[2]//2, size=patch_size)
        self.SEEM = Spectral_Enhance_Module(in_channels=self.planes_b[2], in_channels2=self.planes_a[2])

        self.FusionLayer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.planes_a[2] * 2,
                out_channels=self.planes_a[2],
                kernel_size=1,
            ),
            nn.BatchNorm2d(self.planes_a[2]),
            nn.ReLU(),
        )

        self.fc = nn.Linear(self.planes_a[2], n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        b1_1 = self.branch2(x1)
        b2_1 = self.branch2(x2)

        b1_2 = self.branch3(x1)
        b2_2 = self.branch3(x2)

        b1_3 = x1
        b2_3 = x2

        b1_4 = self.conv1_a(x1)
        b2_4 = self.conv1_b(x2)

        b1_4 = self.conv2_a(b1_4)
        b2_4 = self.conv2_b(b2_4)

        b1_4 = self.conv3_a(b1_4)
        b2_4 = self.conv3_b(b2_4)

        s = (32, 7, 7)

        b1_1 = F.interpolate(b1_1.unsqueeze(0), size=s, mode='nearest').squeeze(0)
        b1_2 = F.interpolate(b1_2.unsqueeze(0), size=s, mode='nearest').squeeze(0)
        b1_3 = F.interpolate(b1_3.unsqueeze(0), size=s, mode='nearest').squeeze(0)

        b2_1 = F.interpolate(b2_1.unsqueeze(0), size=s, mode='nearest').squeeze(0)
        b2_2 = F.interpolate(b2_2.unsqueeze(0), size=s, mode='nearest').squeeze(0)
        b2_3 = F.interpolate(b2_3.unsqueeze(0), size=s, mode='nearest').squeeze(0)

        # print("b1:", b1_1.shape)
        # print("b2:", b1_2.shape)
        # print("b3:", b1_3.shape)
        # print("b4:", b1_4.shape)

        # 归一化权重
        w1_1 = torch.exp(self.w_hsl[0]) / torch.sum(torch.exp(self.w_hsl))
        w1_2 = torch.exp(self.w_hsl[1]) / torch.sum(torch.exp(self.w_hsl))
        w1_3 = torch.exp(self.w_hsl[2]) / torch.sum(torch.exp(self.w_hsl))
        w1_4 = torch.exp(self.w_hsl[3]) / torch.sum(torch.exp(self.w_hsl))

        w2_1 = torch.exp(self.w_lidar[0]) / torch.sum(torch.exp(self.w_lidar))
        w2_2 = torch.exp(self.w_lidar[1]) / torch.sum(torch.exp(self.w_lidar))
        w2_3 = torch.exp(self.w_lidar[2]) / torch.sum(torch.exp(self.w_lidar))
        w2_4 = torch.exp(self.w_hsl[3]) / torch.sum(torch.exp(self.w_hsl))

        x1 = b1_1 * w1_1 + b1_2 * w1_2 + b1_3 * w1_3 + b1_4 * w1_4
        x2 = b2_1 * w2_1 + b2_2 * w2_2 + b2_3 * w2_3 + b2_4 * w2_4

        # print("特征融合结果:", x1.shape)

        ss_x1 = self.SAEM(x1, x2)
        ss_x2 = self.SEEM(x2, x1)
        # print("ss_x1:", ss_x1.shape)
        # print("ss_x2:", ss_x2.shape)
        #x = torch.cat((ss_x1, ss_x2), dim=1)
        # print("x_combine:", x_combine.shape)
        # x = self.branch_SE(x)
        # x = F.interpolate(x.unsqueeze(0), size=s, mode='nearest').squeeze(0)
        # x = self.FusionLayer(x)
        x = self.branch_AFF(ss_x1, ss_x2)
        print("x:", x.shape)
        x = self.avg_pool(x)
        print("x:", x.shape)
        x = torch.flatten(x, 1)
        print("x:", x.shape)
        x = self.fc(x)
        print("x:", x.shape)

        return x


if __name__ == '__main__':
    import torch

    img1 = torch.randn(2, 6, 7, 7)
    img2 = torch.randn(2, 6, 7, 7)

    SAEM = Spatial_Enhance_Module(in_channels=6, inter_channels=6 // 2, size=7)
    out = SAEM(img1, img2)
    print(out)

    SEEM = Spectral_Enhance_Module(in_channels=6, in_channels2=6)
    out = SEEM(img1, img2)
    print(out.shape)