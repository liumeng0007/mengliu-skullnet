import torch
import torch.nn as nn
import timm

# unet encoder + skip connection + channel attention & spatial attention

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dhw):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LayerNorm([dhw, dhw, dhw]),
            # nn.BatchNorm3d(out_channels),
            # nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.LayerNorm([dhw, dhw, dhw]),
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class SkipConnect(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SkipConnect, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        return self.residual(x)


# Attention Mechanism
class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool3d = nn.AdaptiveAvgPool3d(1)  # B*C*1*1*1
        self.max_pool3d = nn.AdaptiveMaxPool3d(1)  # B*C*1*1*1

        self.shared_mlp = nn.Sequential(
            nn.Conv3d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(channel // ratio, channel, 1, bias=False)
        )
        # B*C*1*1*1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.avg_pool3d(x)
        avgout = self.shared_mlp(avgout)
        maxout = self.max_pool3d(x)
        maxout = self.shared_mlp(maxout)
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, channel, kernel_size=7, stride=1, padding=3):
        super(SpatialAttention, self).__init__()
        self.conv3d = nn.Conv3d(2, 1, kernel_size, stride, padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv3d(x))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channel)
        self.spatial_att = SpatialAttention(channel)

    def forward(self, x):
        out = self.channel_att(x) * x
        out = self.spatial_att(out) * out
        return out



class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dwh):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv(in_channels, out_channels, dwh)
        )

    def forward(self, x):
        return self.encoder(x)


class Classifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_features, num_classes)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout()
        # self.fc2 = nn.Linear(out_features, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.drop(self.relu(self.fc1(x)))
        # x = self.drop(self.fc2(x))
        return x


class Encoder1(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, n_channels=16):
        super().__init__()
        self.in_channels = in_channels  # 1
        self.n_classes = n_classes  # 2
        # n_channels -> out_channels
        self.n_channels = n_channels  # 4
        self.conv = DoubleConv(in_channels, n_channels, 224)
        self.skipconnect1 = nn.Sequential(
            nn.Conv3d(in_channels, n_channels, kernel_size=1, stride=1),
            nn.LayerNorm([224, 224, 224])
        )
        self.cbam1 = CBAM(n_channels)

    def forward(self, x):
        x1 = self.conv(x)
        x1 = x1 + self.skipconnect1(x)
        x1 = self.cbam1(x1) + x1
        return x1


class Encoder2(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, n_channels=16):
        super().__init__()
        self.in_channels = in_channels  # 1
        self.n_classes = n_classes  # 2
        # n_channels -> out_channels
        self.n_channels = n_channels  # 4
        self.enc1 = Down(n_channels, 2 * n_channels, 112)
        self.skipconnect2 = SkipConnect(n_channels, 2 * n_channels)
        self.cbam2 = CBAM(2 * n_channels)

    def forward(self, x1):
        x2 = self.enc1(x1)
        x2 = x2 + self.skipconnect2(x1)
        x2 = self.cbam2(x2) + x2
        return x2


class Encoder3(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, n_channels=16):
        super().__init__()
        self.in_channels = in_channels  # 1
        self.n_classes = n_classes  # 2
        # n_channels -> out_channels
        self.n_channels = n_channels  # 4
        self.enc2 = Down(2 * n_channels, 4 * n_channels, 56)
        self.skipconnect3 = SkipConnect(2 * n_channels, 4 * n_channels)
        self.cbam3 = CBAM(4 * n_channels)

    def forward(self, x2):
        x3 = self.enc2(x2)
        x3 = x3 + self.skipconnect3(x2)
        x3 = self.cbam3(x3) + x3
        return x3


class Encoder4(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, n_channels=16):
        super().__init__()
        self.in_channels = in_channels  # 1
        self.n_classes = n_classes  # 2
        # n_channels -> out_channels
        self.n_channels = n_channels  # 4
        self.enc3 = Down(4 * n_channels, 8 * n_channels, 28)
        self.skipconnect4 = SkipConnect(4 * n_channels, 8 * n_channels)
        self.cbam4 = CBAM(8 * n_channels)

    def forward(self, x3):
        x4 = self.enc3(x3)
        x4 = x4 + self.skipconnect4(x3)
        x4 = self.cbam4(x4) + x4
        return x4


class Encoder5(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, n_channels=16):
        super().__init__()
        self.in_channels = in_channels  # 1
        self.n_classes = n_classes  # 2
        # n_channels -> out_channels
        self.n_channels = n_channels  # 4
        self.enc4 = Down(8 * n_channels, 16 * n_channels, 14)
        self.skipconnect5 = SkipConnect(8 * n_channels, 16 * n_channels)
        self.cbam5 = CBAM(16 * n_channels)

    def forward(self, x4):
        x5 = self.enc4(x4)
        x5 = x5 + self.skipconnect5(x4)
        x5 = self.cbam5(x5) + x5
        return x5


class Encoderfinal(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, n_channels=16):
        super().__init__()
        super().__init__()
        self.in_channels = in_channels  # 1
        self.n_classes = n_classes  # 2
        # n_channels -> out_channels
        self.n_channels = n_channels  # 4
        self.encoder1 = Encoder1()
        self.encoder2 = Encoder2()
        self.encoder3 = Encoder3()
        self.encoder4 = Encoder4()
        self.encoder5 = Encoder5()

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)
        return x5


class Transencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.transencoder = timm.models.create_model("vit_small_patch16_224", num_classes=2, in_chans=14)

    def forward(self, x):
        return self.transencoder(x)

class TransNet3d(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, n_channels=16):
        super(TransNet3d, self).__init__()
        self.in_channels = in_channels  # 1
        self.n_classes = n_classes  # 2
        # n_channels -> out_channels
        self.n_channels = n_channels  # 4
        self.encoderfinal = Encoderfinal()
        # self.out = Classifier(512*4*7*7*7, 2)
        self.out = Transencoder()

    def forward(self, x):
        x = self.encoderfinal(x)
        # print(x.shape)
        x = x.reshape((1, -1, 224, 224))
        x = self.out(x)
        return x


class TransNet3d_test(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, n_channels=16):
        super(TransNet3d_test, self).__init__()
        self.in_channels = in_channels  # 1
        self.n_classes = n_classes  # 2
        # n_channels -> out_channels
        self.n_channels = n_channels  # 4

        self.conv = DoubleConv(in_channels, n_channels, 224)
        self.skipconnect1 = nn.Sequential(
            nn.Conv3d(in_channels, n_channels, kernel_size=1, stride=1),
            nn.LayerNorm([224, 224, 224])
        )
        self.cbam1 = CBAM(n_channels)

        self.enc1 = Down(n_channels, 2 * n_channels, 112)
        self.skipconnect2 = SkipConnect(n_channels, 2 * n_channels)
        self.cbam2 = CBAM(2 * n_channels)

        self.enc2 = Down(2 * n_channels, 4 * n_channels, 56)
        self.skipconnect3 = SkipConnect(2 * n_channels, 4 * n_channels)
        self.cbam3 = CBAM(4 * n_channels)

        self.enc3 = Down(4 * n_channels, 8 * n_channels, 28)
        self.skipconnect4 = SkipConnect(4 * n_channels, 8 * n_channels)
        self.cbam4 = CBAM(8 * n_channels)

        self.enc4 = Down(8 * n_channels, 16 * n_channels, 14)
        self.skipconnect5 = SkipConnect(8 * n_channels, 16 * n_channels)
        self.cbam5 =CBAM(16 * n_channels)

        # self.enc5 = Down(16 * n_channels, 32 * n_channels, 7)
        # self.skipconnect6 = SkipConnect(16 * n_channels, 32 * n_channels)
        # self.cbam6 = CBAM(32 * n_channels)

        # self.out = TransEncoder()
        self.out = Classifier(512*4*7*7*7, 2)

    def forward(self, x):
        x1 = self.conv(x)
        x1 = x1 + self.skipconnect1(x)
        x1 = self.cbam1(x1) + x1

        x2 = self.enc1(x1)
        x2 = x2 + self.skipconnect2(x1)
        x2 = self.cbam2(x2) + x2

        x3 = self.enc2(x2)
        x3 = x3 + self.skipconnect3(x2)
        x3 = self.cbam3(x3) + x3

        x4 = self.enc3(x3)
        x4 = x4 + self.skipconnect4(x3)
        x4 = self.cbam4(x4) + x4

        x5 = self.enc4(x4)
        x5 = x5 + self.skipconnect5(x4)
        x5 = self.cbam5(x5) + x5
        #
        # print("x5", x5.shape)

        out = self.out(x5)
        # out = nn.Linear(175616, 2)(x6)
        return out







