import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock2d_Instance(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock2d_Instance, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        y = self.layer(x)
        return y


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock3d, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        y = self.layer(x)
        return y


class AdaNorm3d(nn.Module):
    def __init__(self, num_features, style_dim):
        super(AdaNorm3d, self).__init__()
        self.norm = nn.InstanceNorm3d(num_features=num_features)
        self.style = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        s = self.style(s).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        gamma, beta = s.chunk(2, 1)

        y = self.norm(x)
        y = gamma * y + beta
        return y


class AdaNorm2d(nn.Module):
    def __init__(self, num_features, style_dim):
        super(AdaNorm2d, self).__init__()
        self.norm = nn.InstanceNorm2d(num_features=num_features)
        self.style = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        s = self.style(s).unsqueeze(2).unsqueeze(3)
        gamma, beta = s.chunk(2, 1)

        y = self.norm(x)
        y = gamma * y + beta
        return y


class ConvBlock3d_Ada(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, style_dim=128):
        super(ConvBlock3d_Ada, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.norm1 = AdaNorm3d(num_features=out_channels, style_dim=style_dim)
        self.relu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.norm2 = AdaNorm3d(num_features=out_channels, style_dim=style_dim)
        self.relu2 = nn.LeakyReLU(0.2)

    def forward(self, x, s):
        y = self.conv1(x)
        y = self.norm1(y, s)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.norm2(y, s)
        y = self.relu2(y)

        return y


class ConvBlock2d_Ada(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, style_dim=128):
        super(ConvBlock2d_Ada, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.norm1 = AdaNorm2d(num_features=out_channels, style_dim=style_dim)
        self.relu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.norm2 = AdaNorm2d(num_features=out_channels, style_dim=style_dim)
        self.relu2 = nn.LeakyReLU(0.2)

    def forward(self, x, s):
        y = self.conv1(x)
        y = self.norm1(y, s)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.norm2(y, s)
        y = self.relu2(y)

        return y


class Projection(nn.Module):
    def __init__(self, in_channels=256):
        super(Projection, self).__init__()
        self.layer = nn.Sequential(
            ConvBlock3d(in_channels=in_channels, out_channels=in_channels // 2),
            ConvBlock3d(in_channels=in_channels // 2, out_channels=in_channels // 4)
        )
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels=16 * in_channels // 4, out_channels=1024, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x, rot):
        flow = F.affine_grid(theta=rot, size=x.shape, align_corners=False)
        y = F.grid_sample(x, flow, align_corners=False)

        y = self.layer(y)
        b, c, d, h, w = y.shape
        y = y.view(b, c * d, h, w)
        y = self.proj(y)

        return y