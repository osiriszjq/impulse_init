import torch
import torch.nn as nn
import numpy as np


# initialization of convolusion kernel, C*1*K*K
def SpatialConv2d_init(C, kernel_size, init='random'):
    if (init == 'random')|(init == 'softmax'):
        weight = 1/kernel_size*(2*torch.rand((C,1,kernel_size,kernel_size))-1)
    elif init == 'impulse':
        k = torch.randint(0,kernel_size*kernel_size,(C,1))
        weight = torch.zeros((C,1,kernel_size*kernel_size))
        for i in range(C):
            for j in range(1):
                weight[i,j,k[i,j]] = 1
        weight = np.sqrt(1/3)*weight.reshape(C,1,kernel_size,kernel_size)
    elif init[:3] == 'box':
        weight = torch.zeros((C,1,kernel_size*kernel_size))
        for i in range(C):
            for j in range(1):
                k = np.random.choice(kernel_size*kernel_size,int(init[3:]),replace=False)
                weight[i,j,k] = 1
        weight = np.sqrt(1/int(init[3:])/3)*weight.reshape(C,1,kernel_size,kernel_size)
    elif init[:3] == 'gau':
        k = torch.randint(0,kernel_size,(C,1,2))
        weight = torch.zeros((C,1,kernel_size,kernel_size))
        for i in range(C):
            for j in range(1):
                for p in range(kernel_size):
                    for q in range(kernel_size):
                        weight[i,j,p,q] = (-0.5/float(init[3:])*((p-k[i,j,0])**2+(q-k[i,j,1])**2)).exp()
        weight = weight/((weight.flatten(1,3)**2).sum(1).mean()*3).sqrt()
    else:
        return -1
    return weight


# initialization of convolusion kernel in linear format, C*img_size*img_size, Out of Memory!!!
def SpatialConv2d_Linear_init(C, H, W, init='perm'):
    weight = torch.zeros((C,H*W,H*W))
    if init == 'perm':
        for i in range(C):
            k = torch.randint(0,H*W,(H*W,))
            for j in range(H*W):
                weight[i][j,k[j]] = 1
        weight = np.sqrt(1/3)*weight
    elif init == 'fullperm':
        for i in range(C):
            k = torch.randperm(H*W)
            for j in range(H*W):
                weight[i][j,k[j]] = 1
        weight = np.sqrt(1/3)*weight
    elif init[:6] == 'impulse':
        ff = int(init[7:])
        weight = torch.zeros((C,H*W,H*W))
        k = torch.randint(0,ff**2,(C,))
        for i in range(C):
            m = (k[i]//ff)-(ff//2)
            n = (k[i]%ff)-(ff//2)
            tmp_weight = torch.zeros((W,W))
            for j in range(0-min(0,n),W-max(0,n)):
                tmp_weight[j,j+n] = 1
            for j in range(0-min(0,m),H-max(0,m)):
                weight[i,j*W:(j+1)*W,(j+m)*W:(j+m+1)*W] = tmp_weight
        weight = np.sqrt(1/3)*weight
    else:
        return -1
    return weight



# my spatial conv fuction, group=#channels, heads controls the number of different conv filters
class SpatialConv2d(nn.Module):
    def __init__(self, C, kernel_size, bias=True, init='random', heads = -1, trainable= True, input_weight=None):
        super(SpatialConv2d, self).__init__()
        self.C = C
        self.kernel_size = kernel_size
        self.init = init
        
        # different initialisation
        weight = SpatialConv2d_init(C,kernel_size,init=init)
        
        # how many heads or different filters we want to use
        if (heads<1)|(heads>C) :
            heads = C
        self.choice_idx = np.random.choice(heads,C,replace=(heads<C))

        # if use gloabal weight
        if input_weight is None:
            self.weight = nn.Parameter(weight[:heads],requires_grad=trainable)
        else:
            self.weight = input_weight

        if bias:
            bias = 1/kernel_size*(2*torch.rand((C))-1)
            self.bias = nn.Parameter(bias,requires_grad=trainable)
        else:
            self.bias = None


    def forward(self, x):
        if self.init == 'softmax':
            w_s = self.weight.shape
            return torch.nn.functional.conv2d(x, self.weight.flatten(2,3).softmax(-1).reshape(w_s)[self.choice_idx],self.bias,padding='same',groups=self.C)
        else:
            return torch.nn.functional.conv2d(x, self.weight[self.choice_idx], self.bias,padding='same',groups=self.C)


# my soatial conv function in linear format, out of memory!!!
class SpatialConv2d_LinearFormat(nn.Module):
    def __init__(self, C, H, W, bias=True, init='random', heads=-1, trainable= True, input_weight=None):
        super(SpatialConv2d_LinearFormat, self).__init__()
        self.C = C
        self.H = H
        self.W = W
        self.init = init
        self.heads = heads

        # linear weights initialisation
        weight = SpatialConv2d_Linear_init(C,H,W,init=init)
        
        # how many heads or different filters we want to use
        if (heads<1)|(heads>C) :
            heads = C
        self.choice_idx = np.random.choice(heads,C,replace=(heads<C))
        
        # if use gloabal weight
        if input_weight is None:
            if init[:2] == 'qk':
                self.weight1 = nn.Parameter(weight[0][:heads],requires_grad=trainable)
                self.weight2 = nn.Parameter(weight[1][:heads],requires_grad=trainable)
                self.weight = [self.weight1,self.weight2]
            else:
                self.weight = nn.Parameter(weight[:heads],requires_grad=trainable)
        else:
            self.weight = input_weight

        if bias:
            bias = 1/C*(2*torch.rand((C))-1)
            self.bias = nn.Parameter(bias,requires_grad=trainable)
        else:
            self.bias = None


    def forward(self, x):
        if self.init[:2] == 'qk':
            A = torch.matmul(self.weight[0][self.choice_idx],self.weight[1][self.choice_idx])
        else:
            A = self.weight
        x = torch.matmul(x.reshape(-1,self.C,1,self.H*self.W), A.transpose(-2,-1))
        return (x.reshape(-1,self.C,self.H*self.W)+ self.bias).reshape(-1,self.C,self.H,self.W)


# normal convmixer functions

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=5, patch_size=2, n_classes=10, image_size=32,
                  return_embedding=False, spatial=True, init='random', heads=-1, input_weight=False, spatial_bias=True, linear_format=False):
        super(ConvMixer,self).__init__()
        self.return_embeding = return_embedding
        self.input_embedding = nn.Sequential(nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size), nn.GELU(), nn.BatchNorm2d(dim))
        self.mixer = nn.ModuleList([])

        # use gloabal weights or not
        if input_weight:
            if (heads<1)|(heads>dim) :
                heads = dim
        else:
            self.input_weight = None
        
        # choose spatial conv format
        if linear_format:
            H=int(image_size/patch_size)
            W=int(image_size/patch_size)
            if input_weight:
                self.input_weight = SpatialConv2d_Linear_init(dim, H, W, init=init)
                self.input_weight = nn.Parameter(self.input_weight[:heads],requires_grad=spatial)
            for _ in range(depth):
                self.mixer.append(nn.ModuleList([
                    Residual(nn.Sequential(
                        SpatialConv2d_LinearFormat(dim, H, W, bias=spatial_bias, init=init, heads= heads, trainable=spatial, input_weight=self.input_weight),
                        nn.GELU(), nn.BatchNorm2d(dim))),
                    nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), nn.GELU(), nn.BatchNorm2d(dim))
                ]))
        else:
            if input_weight:
                self.input_weight = SpatialConv2d_init(dim, kernel_size, init=init)
                self.input_weight = nn.Parameter(self.input_weight[:heads],requires_grad=spatial)
            for _ in range(depth):
                self.mixer.append(nn.ModuleList([
                    Residual(nn.Sequential(
                        SpatialConv2d(dim, kernel_size, bias=spatial_bias, init=init, heads= heads, trainable=spatial, input_weight=self.input_weight),
                        nn.BatchNorm2d(dim))),
                        # missing GeLU here !!!!
                    nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1),nn.GELU(),nn.BatchNorm2d(dim))
                ]))

        self.output = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Flatten(),
                                    nn.Linear(dim, n_classes))

    def forward(self, x):
        if self.return_embeding: xs = [x]
        x = self.input_embedding(x)
        if self.return_embeding: xs.append(x)
        for spatial,channel in self.mixer:
            x = spatial(x)
            x = channel(x)
            if self.return_embeding: xs.append(x)
        x = self.output(x)
        if self.return_embeding:
            return x, xs
        else:
            return x