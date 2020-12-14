import torch
import os
import numpy as np
from torch.nn import functional as F
from pytorch_msssim import msssim

def check_dir(dir_path, subdir=None):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if subdir:
        subdir_path = os.path.join(dir_path, subdir)
        if not os.path.exists(subdir_path):
            os.mkdir(subdir_path)

def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)


def mse_loss(recon_x, x, mu, logsigma, _):
    NORM = recon_x.shape[0]*recon_x.shape[1]*recon_x.shape[2]*recon_x.shape[3]
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    BCE = F.mse_loss(recon_x, x, size_average=False)
    #loss per pixel
    loss = (KLD + BCE) / NORM
    return loss, loss.item(), 0.0

def kl_criterion(mu, logvar, batch_size):
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size
  return KLD



def ms_ssim_loss(recon_x, x, _, __):
    loss = 1 - msssim(recon_x, x, normalize=True)

    return loss, 0.0, loss.item()
def mix_loss(recon_x, x, mu, logsigma, alpha):
    NORM = recon_x.shape[0]*recon_x.shape[1]*recon_x.shape[2]*recon_x.shape[3]
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    BCE = F.mse_loss(recon_x, x, reduction='mean')
    msssim_loss = 1.0 - msssim(recon_x, x)

    loss = (KLD/NORM + BCE)*alpha+(1-alpha)*msssim_loss
    return loss, (KLD/NORM + BCE).item()*alpha, (1-alpha)*msssim_loss.item()

