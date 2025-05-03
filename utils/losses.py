import torch
import random


# loss align <-- Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere
# 这个loss计算起来很慢
# bsz : batch size (number of positive pairs)
# d   : latent dim
# x   : Tensor, shape=[bsz, d]
#       latents for one side of positive pairs
# y   : Tensor, shape=[bsz, d]
#       latents for the other side of positive pairs

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def cl_loss(x, y, lam):
    return align_loss(x, y) + lam * (uniform_loss(x) + uniform_loss(y)) / 2

