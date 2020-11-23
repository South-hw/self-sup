import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from lib.blocks import ConvBlock2d_Instance, ConvBlock2d_Ada, ConvBlock3d_Ada, Projection


class Discriminator(nn.Module):
    def __init__(self, style_dim):
        super(Discriminator, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=1, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            ConvBlock2d_Instance(in_channels=128, out_channels=256),
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
            ConvBlock2d_Instance(in_channels=256, out_channels=512),
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
            ConvBlock2d_Instance(in_channels=512, out_channels=512),
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
            ConvBlock2d_Instance(in_channels=512, out_channels=512),
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
            ConvBlock2d_Instance(in_channels=512, out_channels=512),
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.2)
        )

        self.validity_layer = nn.Linear(in_features=512, out_features=1)
        self.style_layer = nn.Linear(in_features=512, out_features=style_dim)
        self.azimuth_layer = nn.Linear(in_features=512, out_features=2)

    def forward(self, x):
        b = x.shape[0]
        x = self.backbone(x).view(b, -1)

        validity = self.validity_layer(x)
        style = (self.style_layer(x)).chunk(2, 1)
        view = self.azimuth_layer(x)
        view = view / torch.norm(view, p=None, dim=1, keepdim=True)

        return validity, style, view


class Generator(nn.Module):
    def __init__(self, style_dim):
        super(Generator, self).__init__()
        self.const_input = nn.Parameter(torch.rand(1, 512, 4, 4, 4))
        self.layer_3d = nn.ModuleList([
            ConvBlock3d_Ada(in_channels=512, out_channels=512, style_dim=style_dim),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ConvBlock3d_Ada(in_channels=512, out_channels=512, style_dim=style_dim),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ConvBlock3d_Ada(in_channels=512, out_channels=256, style_dim=style_dim),
        ])
        self.projection = Projection(in_channels=256)
        self.layer_2d = nn.ModuleList([
            ConvBlock2d_Ada(in_channels=1024, out_channels=512),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock2d_Ada(in_channels=512, out_channels=256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock2d_Ada(in_channels=256, out_channels=128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock2d_Ada(in_channels=128, out_channels=64),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)
        ])

    def forward(self, style3d, style2d, rot):
        b = rot.shape[0]
        y = self.const_input.repeat(b, 1, 1, 1, 1)

        for f in self.layer_3d:
            if isinstance(f, ConvBlock3d_Ada):
                y = f(y, style3d)
            else:
                y = f(y)
        y = self.projection(y, rot)

        for f in self.layer_2d:
            if isinstance(f, ConvBlock2d_Ada):
                y = f(y, style2d)
            else:
                y = f(y)

        return y


class StyleFC(nn.Module):
    def __init__(self, style_dim=128):
        super(StyleFC, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features=style_dim, out_features=style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=style_dim, out_features=style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=style_dim, out_features=style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=style_dim, out_features=style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=style_dim, out_features=style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=style_dim, out_features=style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=style_dim, out_features=style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=style_dim, out_features=style_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, style):
        style = self.layer(style)

        return style


class Synthesizer(nn.Module):
    def __init__(self, style_dim):
        super(Synthesizer, self).__init__()
        self.generator = Generator(style_dim=style_dim)
        self.style_fc = StyleFC(style_dim=style_dim)

    def forward(self, style3d, style2d, rot):
        style3d = self.style_fc(style3d)
        style2d = self.style_fc(style2d)

        out = self.generator(style3d, style2d, rot)
        return out


class AlexPerceptual(nn.Module):
    def __init__(self, root):
        super(AlexPerceptual, self).__init__()
        alex = torchvision.models.alexnet(pretrained=False)
        alex.load_state_dict(torch.load(root))
        self.features = nn.Sequential(
            *list(alex.features.children())[:4]
        )

    def forward(self, x):
        y = self.features(x)
        return y