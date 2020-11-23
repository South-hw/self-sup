import os
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from lib.models import Discriminator, Synthesizer, AlexPerceptual
from lib.utils import rot_azimuth, compute_gradient_penalty
from lib.datasets import H36


def train(gpu, args):
    torch.manual_seed(100)
    rank = args.nr * args.gpus + gpu

    logger = None
    writer = None
    if rank == 0:
        logging.basicConfig(filename=args.logger_root, level=logging.DEBUG)
        logging.getLogger("").addHandler(logging.StreamHandler())
        logger = logging.getLogger(__name__)
        writer = SummaryWriter(args.writer_root)

    process_group = dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')

    discriminator = Discriminator(style_dim=args.style_dim * 2).to(device=device)
    discriminator = DDP(discriminator, device_ids=[gpu])
    discriminator.train()

    generator = Synthesizer(style_dim=args.style_dim).to(device=device)
    generator = DDP(generator, device_ids=[gpu])
    generator.train()

    optimizer_D = torch.optim.Adam(params=discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    optimizer_G = torch.optim.Adam(params=generator.module.generator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    optimizer_G.add_param_group({'params': generator.module.style_fc.parameters(), 'lr': args.lr * 0.01, 'mult': 0.01})

    trsf = transforms.Compose([
        transforms.Resize(size=(128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ds = H36(root=args.root, train=True, range_fr=30, transform=trsf)
    sampler = DistributedSampler(ds, num_replicas=args.world_size, rank=rank)
    dl = DataLoader(dataset=ds, batch_size=args.b_size, shuffle=False, num_workers=4, pin_memory=True, sampler=sampler)

    perceptual_layer = AlexPerceptual(root=args.perceptual_root)
    for epoch in range(args.epochs):
        for i, (x, _) in enumerate(dl):
            b_size = x.shape[0]
            x = x.to(device=device)
            fake_style3d = torch.FloatTensor(2, b_size, args.style_dim).uniform_(-1, 1).to(device=device)
            fake_style2d = torch.FloatTensor(2, b_size, args.style_dim).uniform_(-1, 1).to(device=device)
            fake_view = torch.FloatTensor(2, b_size, 2).uniform_(-1, 1).to(device=device)
            fake_view = fake_view / torch.norm(fake_view, dim=2, keepdim=True)

            # ====================
            # Train Discriminator
            # 1. Image consistency loss
            # 2. Style and View consistency Loss
            # 3. Discriminator Loss
            # ====================
            optimizer_D.zero_grad()

            real_validity, style, view = discriminator(x)
            R = rot_azimuth(v=view)
            x_ = generator(style[0], style[1], R)

            loss_img = 1 - F.cosine_similarity(x_.view(b_size, -1), x.view(b_size, -1))

            fake_R = rot_azimuth(v=fake_view[0])
            fake_x = generator(fake_style3d[0], fake_style2d[0], fake_R)
            fake_validity, fake_style_, fake_view_ = discriminator(fake_x)

            loss_s = (F.mse_loss(fake_style_[0], fake_style3d[0]) + F.mse_loss(fake_style_[1], fake_style2d[1])) * 0.5
            loss_v = ((1 - F.cosine_similarity(fake_view_, fake_view[0])) ** 2).mean()

            gp = compute_gradient_penalty(D=discriminator, real_samples=x, fake_samples=fake_x)
            loss_adv = -real_validity.mean() + fake_validity.mean() + 10 * gp

            loss = loss_img + loss_s + loss_v + loss_adv
            loss.backward()
            optimizer_D.step()

            if writer is not None:
                writer.add_scalar('loss_D/loss', loss.item(), epoch * len(dl) + i)
                writer.add_scalar('loss_D/loss_img', loss_img.item(), epoch * len(dl) + i)
                writer.add_scalar('loss_D/loss_style', loss_s.item(), epoch * len(dl) + i)
                writer.add_scalar('loss_D/loss_view', loss_v.item(), epoch * len(dl) + i)
                writer.add_scalar('loss_D/loss_adv', loss_adv.item(), epoch * len(dl) + i)
                logger.info(f'[{epoch}/{args.epochs}] [{i}/{len(dl)}] loss_D: {loss.item()}    loss_img: {loss_img.item()}'
                            f'    loss_style: {loss_s.item()}    loss_view: {loss_v.item()}    loss_adv: {loss_adv.item()}')
            # ====================
            # Train Generator
            # 1. Discriminator Loss
            # 2. Style and View consistency Loss
            # ====================
            optimizer_G.zero_grad()
            fake_style3d = torch.cat((fake_style3d[0], fake_style3d[0], fake_style3d[1]), dim=0)
            fake_style2d = torch.cat((fake_style2d[0], fake_style2d[0], fake_style2d[1]), dim=0)
            fake_view = torch.cat((fake_view[0], fake_view[1], fake_view[1]), dim=0)

            fake_R = rot_azimuth(v=fake_view)
            fake_x = generator(fake_style3d, fake_style2d, fake_R)
            fake_validity, fake_style_, fake_view_ = discriminator(fake_x)

            loss_adv = -fake_validity.mean()
            loss_s = (F.mse_loss(fake_style_[0], fake_style3d) + F.mse_loss(fake_style_[1], fake_style2d)) * 0.5
            loss_v = ((1 - F.cosine_similarity(fake_view_, fake_view)) ** 2).mean()

            loss = loss_adv + loss_s + loss_v
            loss.backward()
            optimizer_G.step()

            if writer is not None:
                writer.add_scalar('loss_G/loss', loss.item(), epoch * len(dl) + i)
                writer.add_scalar('loss_G/loss_adv', loss_adv.item(), epoch * len(dl) + i)
                writer.add_scalar('loss_G/loss_style', loss_s.item(), epoch * len(dl) + i)
                writer.add_scalar('loss_G/loss_view', loss_v.item(), epoch * len(dl) + i)
                logger.info(f'[{epoch}/{args.epochs}] [{i}/{len(dl)}] loss_G: {loss.item()}'
                            f'loss_style: {loss_s.item()}    loss_view: {loss_v.item()}    loss_adv: {loss_adv.item()}')
        if rank == 0:
            torch.save(discriminator.module.state_dict(), f'../output/weights/discriminator{epoch}.pth')
            torch.save(generator.module.state_dict(), f'../output/weights/generator{epoch}.pth')
            logger.info("=============== SAVE ===============")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', default=1, type=int, help='the number of gpu ndoes')
    parser.add_argument('--gpus', default=1, type=int, help='the number of gpus per node')
    parser.add_argument('--nr', default=0, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epochs', default=50, type=float)
    parser.add_argument('--b_size', default=4, type=int)
    parser.add_argument('--data_root', default='/home/nam/data/human36/subject/upright_images', type=str)
    parser.add_argument('--style_dim', default=128, type=int)
    parser.add_argument('--perceptual_root', default='/home/nam/data/perceptual/alexnet.pth', type=str)
    parser.add_argument('--writer_root', default='/home/nam/research/self_sup/output/tensorboard/test')
    parser.add_argument('--logger_root', default='../output/log/temp.log', type=str)
    args = parser.parse_args()

    args.world_size = args.nodes * args.gpus

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    mp.spawn(train, nprocs=args.gpus, arg=(args, ))


if __name__ == '__main__':
    main()