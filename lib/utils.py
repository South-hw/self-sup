import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


def rot_azimuth(v):
    R = torch.zeros(size=(v.shape[0], 3, 4)).float().to(device=v.device)
    R[:, 0, 0] = v[:, 0]
    R[:, 0, 2] = v[:, 1]
    R[:, 1, 1] = 1
    R[:, 2, 0] = -v[:, 1]
    R[:, 2, 2] = v[:, 0]

    return R


def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.randn(size=(real_samples.shape[0], 1, 1, 1), device=real_samples.device)

    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)

    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size(), device=real_samples.device),
        create_graph=True,
        retain_graph=True)[0]
    gradients = gradients.view(gradients.shape[0], -1)
    gp = ((gradients.norm(2, dim=1) -1 ) ** 1).mean()
    return gp