import torch
import torch.nn as nn
from torch.nn.modules import conv, Linear
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
import numpy as np
import math

def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input_):
        x = input_   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        output = (cos_theta, phi_theta)
        return output # size=(B,Classnum,2)

class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)

class Generator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers_0 = []
        layers_1 = []

        layers_0.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers_0.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers_0.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers_0.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers_0.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers_0.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        self.encoder = nn.Sequential(*layers_0)

        # curr_dim + c_dim -> curr_dim
        layers_1.append(nn.Conv2d(curr_dim+c_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False))
        layers_1.append(nn.InstanceNorm2d(curr_dim, affine=True))
        layers_1.append(nn.ReLU(inplace=True))

        # Bottleneck
        for i in range(repeat_num):
            layers_1.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        self.bottle_out = nn.Sequential(*layers_1)

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*layers)

    def forward(self, x, c):
        
        x = self.encoder(x)

        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        return self.decoder(self.bottle_out(x))


class Generator_gpb(nn.Module):
    """Generator. Encoder-Decoder Architecture. Add a global pooling branch at the output of the encoder"""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator_gpb, self).__init__()

        layers = []
        layers_0 = []
        layers_1 = []

        layers_0.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers_0.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers_0.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers_0.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers_0.append(nn.InstanceNorm2d(curr_dim*2, affine=True))

            # gpb1
            # layers_0.append(nn.ReLU(inplace=True))

            # gpb2
            if i!=1: # no ReLU at the last down-sampling layer
                layers_0.append(nn.ReLU(inplace=True))

            curr_dim = curr_dim * 2

        self.encoder = nn.Sequential(*layers_0)

        self.gpb = nn.AvgPool2d(44) # 44 is a fixd number for the 176 input size

        # curr_dim + c_dim -> curr_dim
        layers_1.append(nn.Conv2d(curr_dim+c_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False))
        layers_1.append(nn.InstanceNorm2d(curr_dim, affine=True))
        layers_1.append(nn.ReLU(inplace=True))

        # Bottleneck
        for i in range(repeat_num):
            layers_1.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        self.bottle_out = nn.Sequential(*layers_1)

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*layers)

    def forward(self, x, c):
        
        x = self.encoder(x)
        id_vector = torch.squeeze(self.gpb(x))
        id_vector = id_vector/torch.norm(id_vector, p=2, dim=1, keepdim=True)

        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        return self.decoder(self.bottle_out(x)), id_vector


'''
vanilla
'''
class Discriminator(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=k_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_real = self.conv1(h)
        out_aux = self.conv2(h)
        return out_real.squeeze(), out_aux.squeeze()

'''
+ SN
'''
class Discriminator_SN(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator_SN, self).__init__()

        layers = []
        layers.append(SNConv2d(3, conv_dim, 4, 2, 1, bias=True))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(SNConv2d(curr_dim, curr_dim*2, 4, 2, 1, bias=True))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = SNConv2d(curr_dim, 1, 3, 1, 1, bias=False)
        self.conv2 = SNConv2d(curr_dim, c_dim, k_size, 1, 0, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_real = self.conv1(h)
        out_aux = self.conv2(h)
        return out_real.squeeze(), out_aux.squeeze()

'''
+ identity classification w/ cross-entropy
'''
class Discriminator_idcls_cross(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6, classnum=294, feature=False):
        super(Discriminator_idcls_cross, self).__init__()

        self.feature = feature
        self.classnum = classnum
        self.c_dim = c_dim

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        # layers.append(SNConv2d(3, conv_dim, 4, 2, 1, bias=True))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            # layers.append(SNConv2d(curr_dim, curr_dim*2, 4, 2, 1, bias=True))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))

        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv1 = SNConv2d(curr_dim, 1, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, self.c_dim+512, kernel_size=k_size, bias=False)
        # self.conv2 = SNConv2d(curr_dim, self.c_dim+512, k_size, 1, 0, bias=False)
        self.fc = nn.Linear(512, self.classnum)

    def forward(self, x):
        h = self.main(x)
        out_real = self.conv1(h)
        out_aux = self.conv2(h)

        out_feature = out_aux[:, -512:, :, :].contiguous()
        out_aux = out_aux[:, :self.c_dim, :, :]
        
        out_feature = out_feature.view(-1, 512)

        if self.feature:
            return out_real.squeeze(), out_aux.squeeze(), out_feature.squeeze()

        return out_real.squeeze(), out_aux.squeeze(), self.fc(out_feature)

'''
+ identity classification w/ angle loss
'''
class Discriminator_idcls_angle(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6, classnum=294, feature=False):
        super(Discriminator_idcls_angle, self).__init__()

        self.feature = feature
        self.classnum = classnum
        self.c_dim = c_dim

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        # layers.append(SNConv2d(3, conv_dim, 4, 2, 1, bias=True))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            # layers.append(SNConv2d(curr_dim, curr_dim*2, 4, 2, 1, bias=True))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))

        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv1 = SNConv2d(curr_dim, 1, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, self.c_dim+512, kernel_size=k_size, bias=False)
        # self.conv2 = SNConv2d(curr_dim, self.c_dim+512, k_size, 1, 0, bias=False)
        self.fc = AngleLinear(512, self.classnum)

    def forward(self, x):
        h = self.main(x)
        out_real = self.conv1(h)
        out_aux = self.conv2(h)

        out_feature = out_aux[:, -512:, :, :].contiguous()
        out_aux = out_aux[:, :self.c_dim, :, :]
        
        out_feature = out_feature.view(-1, 512)

        if self.feature:
            return out_real.squeeze(), out_aux.squeeze(), out_feature.squeeze()

        return out_real.squeeze(), out_aux.squeeze(), self.fc(out_feature)

'''
+ identity classification w/ cross-entropy
+ SN
'''
class Discriminator_idcls_cross_SN(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6, classnum=294, feature=False):
        super(Discriminator_idcls_cross_SN, self).__init__()

        self.feature = feature
        self.classnum = classnum
        self.c_dim = c_dim

        layers = []
        # layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(SNConv2d(3, conv_dim, 4, 2, 1, bias=True))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            # layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(SNConv2d(curr_dim, curr_dim*2, 4, 2, 1, bias=True))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))

        self.main = nn.Sequential(*layers)
        # self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = SNConv2d(curr_dim, 1, 3, 1, 1, bias=False)
        # self.conv2 = nn.Conv2d(curr_dim, self.c_dim+512, kernel_size=k_size, bias=False)
        self.conv2 = SNConv2d(curr_dim, self.c_dim+512, k_size, 1, 0, bias=False)
        self.fc = SNLinear(512, self.classnum)

    def forward(self, x):
        h = self.main(x)
        out_real = self.conv1(h)
        out_aux = self.conv2(h)

        out_feature = out_aux[:, -512:, :, :].contiguous()
        out_aux = out_aux[:, :self.c_dim, :, :]
        
        out_feature = out_feature.view(-1, 512)

        if self.feature:
            return out_real.squeeze(), out_aux.squeeze(), out_feature.squeeze()

        return out_real.squeeze(), out_aux.squeeze(), self.fc(out_feature)

'''
+ identity classification w/ angle loss
+ SN
'''
class Discriminator_idcls_angle_SN(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6, classnum=294, feature=False):
        super(Discriminator_idcls_angle_SN, self).__init__()

        self.feature = feature
        self.classnum = classnum
        self.c_dim = c_dim

        layers = []
        # layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(SNConv2d(3, conv_dim, 4, 2, 1, bias=True))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            # layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(SNConv2d(curr_dim, curr_dim*2, 4, 2, 1, bias=True))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))

        self.main = nn.Sequential(*layers)
        # self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = SNConv2d(curr_dim, 1, 3, 1, 1, bias=False)
        # self.conv2 = nn.Conv2d(curr_dim, self.c_dim+512, kernel_size=k_size, bias=False)

        self.conv2 = SNConv2d(curr_dim, self.c_dim+512, k_size, 1, 0, bias=False)
        self.fc = AngleLinear(512, self.classnum)

    def forward(self, x):
        h = self.main(x)

        out_real = self.conv1(h)
        out_aux = self.conv2(h)

        out_feature = out_aux[:, -512:, :, :].contiguous()
        out_aux = out_aux[:, :self.c_dim, :, :]
        
        out_feature = out_feature.view(-1, 512)

        if self.feature:
            return out_real.squeeze(), out_aux.squeeze(), out_feature.squeeze()

        return out_real.squeeze(), out_aux.squeeze(), self.fc(out_feature)

#define a Res Generator
def _l2normalize(v, eps=1e-12):
    return v / (((v**2).sum())**0.5 + eps)

def max_singular_value(W, u=None, Ip=1):
    """
    power iteration for weight parameter
    """
    #xp = W.data
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0, 1).cuda()
    _u = u
    for _ in range(Ip):
        #print(_u.size(), W.size())
        _v = _l2normalize(torch.matmul(_u, W.data), eps=1e-12)
        _u = _l2normalize(torch.matmul(_v, torch.transpose(W.data, 0, 1)), eps=1e-12)
    sigma = torch.matmul(torch.matmul(_v, torch.transpose(W.data, 0, 1)), torch.transpose(_u, 0, 1))
    #sigma = torch.sum(_u * torch.transpose(torch.matmul(W.data, torch.transpose(_v, 0, 1)), 0, 1), 1)
    return sigma, _u

class SNConv2d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SNConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)
        self.u = None

    def forward(self, input):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u = _u
        self.weight.data = self.weight.data / sigma
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class SNLinear(Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SNLinear, self).__init__(in_features, out_features, bias)
        self.u = None
    def forward(self, input):
        w_mat = self.weight
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u = _u
        self.weight.data = self.weight.data / sigma
        return F.linear(input, self.weight, self.bias)