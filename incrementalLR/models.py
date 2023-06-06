import torch
import torch.nn.functional as F
import torch.nn as nn
from .default import _C as cfg


class Norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.normalize(x)


def weights_init_mnist_model(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # print(classname)
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # print(classname)
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class dcganDiscriminator(nn.Module):
    def __init__(self, nc=3, nz=128, ndf=64):
        super(dcganDiscriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, nz, 4, 1, 0, bias=False),
            nn.Flatten()
        )

    def forward(self, input):
        return F.normalize(self.main(input))


# Generator
class dcganGenerator(nn.Module):
    def __init__(self, nc=3, nz=128, ngf=64):
        super(dcganGenerator, self).__init__()
        # self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)




def get_model():
    if cfg.MODEL.BACKBONE == 'dcgan':
        print("building the dcgan model...")
        netG = dcganGenerator(nc=cfg.MODEL.NC, nz=cfg.MODEL.NZ, ngf=cfg.MODEL.NGF)
        netD = dcganDiscriminator(nc=cfg.MODEL.NC, nz=cfg.MODEL.NZ, ndf=cfg.MODEL.NDF)
    else:
        raise ValueError()

    netG = nn.DataParallel(netG.cuda()).eval()
    netD = nn.DataParallel(netD.cuda()).eval()

    return netG, netD
