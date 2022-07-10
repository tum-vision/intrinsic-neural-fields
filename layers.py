import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatEnc(nn.Module):
    """
    Inspired by
    https://github.com/facebookresearch/pytorch3d/blob/fc4dd80208bbcf6f834e7e1594db0e106961fd60/pytorch3d/renderer/implicit/harmonic_embedding.py#L10
    """
    def __init__(self, k, include_input=True, use_logspace=False, max_freq=None):
        super(FourierFeatEnc, self).__init__()
        if use_logspace:
            freq_bands = 2 ** torch.arange(0, k) * torch.pi
        else:
            assert max_freq is not None
            freq_bands = 2 ** torch.linspace(0, max_freq, steps=k+1)[:-1] * torch.pi
        self.register_buffer("freq_bands", freq_bands, persistent=False)
        self.include_input = include_input

    def forward(self, x):
        embed = (x[..., None] * self.freq_bands).view(*x.size()[:-1], -1)
        if self.include_input:
            return torch.cat((embed.cos(), embed.sin(), x), dim=-1)
        return torch.cat((embed.cos(), embed.sin()), dim=-1)


class RandomFourierFeatEnc(nn.Module):
    def __init__(self, k, std=1., in_dim=3, dtype=torch.float32, include_input=True):
        super(RandomFourierFeatEnc, self).__init__()
        B = torch.randn((in_dim, k), dtype=dtype) * std
        self.register_buffer("B", B, persistent=True)
        self.include_input = include_input

    def forward(self, x):
        embed = (2 * torch.pi * x) @ self.B
        if self.include_input:
            return torch.cat((embed.cos(), embed.sin(), x), dim=-1)
        return torch.cat((embed.cos(), embed.sin()), dim=-1)
    
    
class Sine(nn.Module):
    def __init__(self):
        super(Sine, self).__init__()
        
    def forward(self, x):
        return torch.sin(x)


class LinearWithConcatAndActivation(nn.Module):
    def __init__(self, x_in_dim, y_in_dim, out_dim, batchnorm=False, activation=nn.ReLU):
        super(LinearWithConcatAndActivation, self).__init__()
        self.Lx = nn.Linear(x_in_dim, out_dim)
        self.Ly = nn.Linear(y_in_dim, out_dim)
        self.actn = activation()
        self.batchnorm = None
        if batchnorm:
            self.batchnorm = nn.BatchNorm1d(out_dim)

    def forward(self, x, y):
        out = self.actn(self.Lx(x) + self.Ly(y))
        return out if self.batchnorm is None else self.batchnorm(out)


class MLP(nn.Module):
    def __init__(self, 
                 in_channels, 
                 hidden_channels, 
                 out_channels, 
                 num_layers, 
                 use_bn=False, 
                 use_ln=False, 
                 dropout=0.5, 
                 activation='relu', 
                 residual=False):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        if use_bn: self.bns = nn.ModuleList()
        if use_ln: self.lns = nn.ModuleList()
        
        if num_layers == 1:
            # linear mapping
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
            if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            for layer in range(num_layers-2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
                if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Invalid activation')
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.dropout = dropout
        self.residual = residual
            
    def forward(self, x):
        x_prev = x
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.activation(x)
            if self.use_bn:
                if x.ndim == 2:
                    x = self.bns[i](x)
                elif x.ndim == 3:
                    x = self.bns[i](x.transpose(2,1)).transpose(2,1)
                else:
                    raise ValueError('invalid dimension of x')
            if self.use_ln: x = self.lns[i](x)
            if self.residual and x_prev.shape == x.shape: x = x + x_prev
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_prev = x
        x = self.lins[-1](x)
        if self.residual and x_prev.shape == x.shape:
            x = x + x_prev
        return x
