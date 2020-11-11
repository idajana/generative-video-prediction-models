import torch
import torch.nn as nn
from torch.autograd import Variable

import pdb
class VAE(nn.Module):
    def __init__(self, nc, ngf, ndf, latent_variable_size, cuda):
        super(VAE, self).__init__()
        self.cuda = cuda
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv2d(nc, ndf, 4, stride=2)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2 = nn.Conv2d(ndf, ndf*2, 4, stride=2)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        self.e3 = nn.Conv2d(ndf*2, ndf*4, 4, stride=2)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        self.e4 = nn.Conv2d(ndf*4, ndf*8, 4, stride=2)
        self.bn4 = nn.BatchNorm2d(ndf*8)


        self.fc1 = nn.Linear(ndf*8*4, latent_variable_size)
        self.fc2 = nn.Linear(ndf*8*4, latent_variable_size)
        # decoder
        self.d1 = nn.Linear(latent_variable_size, ngf*8*4)

        self.deconv1 = nn.ConvTranspose2d(ngf*8*4, 4*ngf, 5, stride=2)
        self.bn5 = nn.BatchNorm2d(ngf*4, 1.e-3)
        self.deconv2 = nn.ConvTranspose2d(4*ngf, 2*ngf, 5, stride=2)
        self.bn6 = nn.BatchNorm2d(ngf*2, 1.e-3)
        self.deconv3 = nn.ConvTranspose2d(ngf*2, ngf, 6, stride=2)
        self.bn7 = nn.BatchNorm2d(ngf, 1.e-3)
        self.deconv4 = nn.ConvTranspose2d(ngf, self.nc, 6, stride=2)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        #h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h4 = h4.view(-1, self.ndf*8*4)
        #return mu, logvar
        return self.fc1(h4), self.fc2(h4)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):

        h1 = self.relu(self.d1(z))
        h1 = h1.unsqueeze(-1).unsqueeze(-1)
        h1 = self.leakyrelu(self.bn5(self.deconv1(h1)))
        h1 = self.leakyrelu(self.bn6(self.deconv2(h1)))
        h1 = self.leakyrelu(self.bn7(self.deconv3(h1)))
        #reconstruction
        return self.sigmoid(self.deconv4(h1))

    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)

        return z

    def forward(self, x):

        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)

        return res, mu, logvar