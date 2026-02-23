import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1  # 输出通道数扩展倍数

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 如果需要下采样（调整维度），则使用下采样层
        self.downsample = downsample

    def forward(self, x):
        identity = x  # 保存输入用于快捷连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果需要下采样，调整identity的维度
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 快捷连接
        out = self.relu(out)

        return out

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入和输出通道数不匹配，需要调整残差连接中的通道数
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)  # 残差连接
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # 残差相加
        out = self.relu(out)
        return out

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out

# 定义上采样块，使用转置卷积进行上采样
class ResUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ResBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 拼接编码器特征
        x = torch.cat([x2, x1], dim=1)
        return self.conv_block(x)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3_6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.conv3_12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.conv3_18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        )
        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Different scales with atrous convolutions
        out1 = self.conv1(x)
        out2 = self.conv3_6(x)
        out3 = self.conv3_12(x)
        out4 = self.conv3_18(x)
        out5 = self.global_avg_pool(x)
        out5 = F.interpolate(out5, size=x.shape[2:], mode='bilinear', align_corners=True)  # Upsample

        out = torch.cat([out1, out2, out3, out4, out5], dim=1)  # Concatenate
        out = self.conv_out(out)
        out = self.bn(out)
        return self.relu(out)

# 定义CBAM模块
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # 通道注意力机制
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 空间注意力机制
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # 通道注意力
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_attention = self.sigmoid(avg_out + max_out)
        x = x * channel_attention

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.sigmoid(self.spatial_conv(spatial_attention))

        x = x * spatial_attention
        return x

class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


# 定义上采样块，使用转置卷积进行上采样
class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = UNetConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 拼接编码器特征
        x = torch.cat([x2, x1], dim=1)
        return self.conv_block(x)

# 定义完整的 U-Net 模型
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # 编码器部分 (下采样)
        self.enc1 = UNetConvBlock(3, 64)
        self.enc2 = UNetConvBlock(64, 128)
        self.enc3 = UNetConvBlock(128, 256)
        self.enc4 = UNetConvBlock(256, 512)

        # 中间部分
        self.middle = UNetConvBlock(512, 1024)

        # 解码器部分 (上采样)
        self.up4 = UNetUpBlock(1024, 512)
        self.up3 = UNetUpBlock(512, 256)
        self.up2 = UNetUpBlock(256, 128)
        self.up1 = UNetUpBlock(128, 64)

        # 最后的 1x1 卷积，用于将输出通道数变为 1
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, kernel_size=2))
        e3 = self.enc3(F.max_pool2d(e2, kernel_size=2))
        e4 = self.enc4(F.max_pool2d(e3, kernel_size=2))

        # 中间部分
        middle = self.middle(F.max_pool2d(e4, kernel_size=2))

        # 解码器
        d4 = self.up4(middle, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        # 最后的输出层
        out = torch.sigmoid(self.final(d1))  # 输出范围在 [0, 1]
        return out
    
class ResUNet2(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2,2,2], num_classes=9):
        super(ResUNet2, self).__init__()
        self.num_classes = num_classes

        # 编码器部分 (下采样)
        self.enc1 = ResBlock(3, 32)
        self.enc2 = self._make_layer(block, 32, 64, layers[0], stride=2)
        self.enc3 = self._make_layer(block, 64, 128, layers[1], stride=2)
        self.enc4 = self._make_layer(block, 128, 256, layers[2], stride=2)

        # 中间部分
        self.middle = ASPP(256, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(64, self.num_classes))

        # 解码器部分 (上采样)
        self.up4 = UNetUpBlock(512, 256)
        self.up3 = UNetUpBlock(256, 128)
        self.up2 = UNetUpBlock(128, 64)
        self.up1 = UNetUpBlock(64, 32)

        # 最后的 1x1 卷积，用于将输出通道数变为 1
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # 中间部分+预测部分
        middle = self.middle(F.max_pool2d(e4, kernel_size=2))
        pred = self.avgpool(middle)
        pred = torch.flatten(pred, 1)
        pred = self.fc(pred)

        # 解码器
        d4 = self.up4(middle, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        # 最后的输出层
        out = torch.sigmoid(self.final(d1))  # 输出范围在 [0, 1]
        return out, pred

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        # 如果输入和输出的维度不同，或者步幅不为1，需要下采样
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # 第一个残差块可能需要下采样
        layers.append(block(in_channels, out_channels, stride, downsample))
        # 其余的残差块
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)


class ResUNet2Dist(nn.Module):
    """ResUNet2 with an extra distance map head."""

    def __init__(self, block=BasicBlock, layers=[2, 2, 2], num_classes=9):
        super(ResUNet2Dist, self).__init__()
        self.num_classes = num_classes

        # 编码器部分 (下采样)
        self.enc1 = ResBlock(3, 32)
        self.enc2 = self._make_layer(block, 32, 64, layers[0], stride=2)
        self.enc3 = self._make_layer(block, 64, 128, layers[1], stride=2)
        self.enc4 = self._make_layer(block, 128, 256, layers[2], stride=2)

        # 中间部分
        self.middle = ASPP(256, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(64, self.num_classes),
        )

        # 解码器部分 (上采样)
        self.up4 = UNetUpBlock(512, 256)
        self.up3 = UNetUpBlock(256, 128)
        self.up2 = UNetUpBlock(128, 64)
        self.up1 = UNetUpBlock(64, 32)

        # 分割输出与距离图输出
        self.final = nn.Conv2d(32, 1, kernel_size=1)
        self.dist_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # 中间部分+预测部分
        middle = self.middle(F.max_pool2d(e4, kernel_size=2))
        pred = self.avgpool(middle)
        pred = torch.flatten(pred, 1)
        pred = self.fc(pred)

        # 解码器
        d4 = self.up4(middle, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        seg = torch.sigmoid(self.final(d1))
        dist = torch.sigmoid(self.dist_head(d1))
        return seg, pred, dist

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)
    
class ResUNet1(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2,2,2], num_classes=6):
        super(ResUNet1, self).__init__()
        self.num_classes = num_classes

        # 编码器部分 (下采样)
        self.enc1 = ResBlock(3, 32)
        self.enc2 = self._make_layer(block, 32, 64, layers[0], stride=2)
        self.enc3 = self._make_layer(block, 64, 128, layers[1], stride=2)
        self.enc4 = self._make_layer(block, 128, 256, layers[2], stride=2)

        # 中间部分
        self.middle = ASPP(256, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, self.num_classes))

        # 解码器部分 (上采样)
        self.up4 = ResUpBlock(512, 256)
        self.up3 = ResUpBlock(256, 128)
        self.up2 = ResUpBlock(128, 64)
        self.up1 = ResUpBlock(64, 32)

        # 最后的 1x1 卷积，用于将输出通道数变为 1
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # 中间部分+预测部分
        middle = self.middle(F.max_pool2d(e4, kernel_size=2))
        pred = self.avgpool(middle)
        pred = torch.flatten(pred, 1)
        pred = self.fc(pred)

        # 解码器
        d4 = self.up4(middle, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        # 最后的输出层
        out = torch.sigmoid(self.final(d1))  # 输出范围在 [0, 1]
        return out, pred
    
    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        # 如果输入和输出的维度不同，或者步幅不为1，需要下采样
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # 第一个残差块可能需要下采样
        layers.append(block(in_channels, out_channels, stride, downsample))
        # 其余的残差块
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)
    
class ResUNet(nn.Module):
    def __init__(self):
        super(ResUNet, self).__init__()

        # 编码器部分 (下采样)
        self.enc1 = ResBlock(3, 64)
        self.enc2 = ResBlock(64, 128)
        self.enc3 = ResBlock(128, 256)
        self.enc4 = ResBlock(256, 512)

        # 中间部分+预测部分
        self.middle = ASPP(512, 1024)

        # 解码器部分 (上采样)
        self.up4 = ResUpBlock(1024, 512)
        self.up3 = ResUpBlock(512, 256)
        self.up2 = ResUpBlock(256, 128)
        self.up1 = ResUpBlock(128, 64)

        # 最后的 1x1 卷积，用于将输出通道数变为 1
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, kernel_size=2))
        e3 = self.enc3(F.max_pool2d(e2, kernel_size=2))
        e4 = self.enc4(F.max_pool2d(e3, kernel_size=2))

        # 中间部分
        middle = self.middle(F.max_pool2d(e4, kernel_size=2))

        # 解码器
        d4 = self.up4(middle, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        # 最后的输出层
        out = torch.sigmoid(self.final(d1))  # 输出范围在 [0, 1]
        return out
    
class CBAMUNet(nn.Module):
    def __init__(self):
        super(CBAMUNet, self).__init__()

        # 编码器部分（使用残差块和CBAM）
        self.enc1 = ResBlock(3, 64)
        self.cbam1 = CBAM(64)  # CBAM 注意力模块
        self.enc2 = ResBlock(64, 128)  # 使用步幅为2的卷积代替池化
        self.cbam2 = CBAM(128)
        self.enc3 = ResBlock(128, 256)
        self.cbam3 = CBAM(256)
        self.enc4 = ResBlock(256, 512)
        self.cbam4 = CBAM(512)

        # 中间的卷积层
        self.middle = ASPP(512, 1024)

        # 解码器部分 (上采样)
        self.up4 = UNetUpBlock(1024, 512)
        self.up3 = UNetUpBlock(512, 256)
        self.up2 = UNetUpBlock(256, 128)
        self.up1 = UNetUpBlock(128, 64)

        # 最终的输出层
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):

        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(self.cbam1(F.max_pool2d(e1, kernel_size=2)))
        e3 = self.enc3(self.cbam2(F.max_pool2d(e2, kernel_size=2)))
        e4 = self.enc4(self.cbam3(F.max_pool2d(e3, kernel_size=2)))

        # 中间部分
        middle = self.middle(self.cbam4(F.max_pool2d(e4, kernel_size=2)))

        # 解码器
        d4 = self.up4(middle, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        # 最后的输出层
        out = torch.sigmoid(self.final(d1))  # 输出范围在 [0, 1]
        return out

# 示例用法
if __name__ == "__main__":
    model = ResUNet1(BasicBlock)
    # print(model)

    # 测试输入
    x = torch.randn(1, 3, 800, 800)  # (batch_size, channels, height, width)
    out, pred = model(x)
    print(out.shape)  # 输出形状应该是 (1, 1, 256, 256)
    print(pred)
