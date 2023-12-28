import torch
from torch import nn
import torch.optim as optim

import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


# my impulse initilization function
def impulse_init(heads,img_size,att_rank,ff,scale=1.0,spatial_pe=None,norm=1):
    weight = torch.zeros((heads,img_size**2,img_size**2))
    k = torch.randint(0,ff**2,(heads,))
    for i in range(heads):
        m = (k[i]//ff)-(ff//2)
        n = (k[i]%ff)-(ff//2)
        tmp_weight = torch.zeros((img_size,img_size))
        for j in range(0-min(0,n),img_size-max(0,n)):
            tmp_weight[j,j+n] = 1
        for j in range(0-min(0,m),img_size-max(0,m)):
            weight[i,j*img_size:(j+1)*img_size,(j+m)*img_size:(j+m+1)*img_size] = tmp_weight
    # weight = np.sqrt(1/3)*weight
    class PermuteM(nn.Module):
        def __init__(self, heads, img_size, att_rank,scale=1.0,spatial_pe=None):
            super().__init__()
            self.scale = scale
            if spatial_pe is None:
                self.spatial_pe = False
                weights_Q = np.sqrt(1/att_rank/heads)*(2*torch.rand(heads,img_size,att_rank)-1)
                weights_K = np.sqrt(1/att_rank/heads)*(2*torch.rand(heads,att_rank,img_size)-1)
            else:
                self.spatial_pe = True
                self.pe = spatial_pe.cuda()
                weights_Q = np.sqrt(1/att_rank/heads)*(2*torch.rand(heads,spatial_pe.shape[1],att_rank)-1)
                weights_K = np.sqrt(1/att_rank/heads)*(2*torch.rand(heads, att_rank, spatial_pe.shape[1])-1)

            self.weights_K = nn.Parameter(weights_K)
            self.weights_Q = nn.Parameter(weights_Q)
        def forward(self):
            if self.spatial_pe:
                M = self.pe@self.weights_Q@self.weights_K@(self.pe.T)
            else:
                M = torch.bmm(self.weights_Q,self.weights_K)
            return torch.softmax(M*self.scale,-1)
    
    net = PermuteM(heads,img_size**2,att_rank,scale,spatial_pe)
    net.cuda()

    nq = net.weights_Q.detach().cpu().norm(dim=(1)).mean()
    weight = weight.cuda()
    num_epoch = 10000
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)#,weight_decay=1e-6)
    for i in range(num_epoch):
        if i%norm==0:
            with torch.no_grad():
                net.weights_Q.div_(net.weights_Q.detach().norm(dim=(1),keepdim=True)/nq)
                net.weights_K.div_(net.weights_K.detach().norm(dim=(1),keepdim=True)/nq)
        optimizer.zero_grad()
        outputs = net()
        loss = criterion(outputs, weight)
        loss.backward()
        optimizer.step()
    print(loss.data)

    return net.weights_Q.detach().cpu(),net.weights_K.detach().cpu()

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, use_value = True, spatial_pe = None, 
                 spatial_x = True, init = 'none', alpha=1.0, trainable=True, out_layer=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)

        self.alpha = alpha

        # input to q&k
        self.spatial_x = spatial_x
        self.spatial_pe = False
        if spatial_pe is not None:
            self.spatial_pe = True
            self.pos_embedding = spatial_pe

        # format & initilization of q&k
        self.init = init
        if init == 'none':
            self.to_qk = nn.Linear(dim, inner_dim*2, bias = False)
        else:
            if init[:7] == 'impulse':
                a, b, c, d, e = init[7:].split('_')
                img_size = int(a)
                att_rank = int(b)
                ff = int(c)
                self.scale = float(d)
                norm = int(e)
                Q, K = impulse_init(heads,img_size,att_rank,ff,self.scale,spatial_pe,norm)
            elif init[:6] == 'random':
                a, b = init[6:].split('_')
                img_size = int(a)
                att_rank = int(b)
                Q = np.sqrt(1/img_size)*(2*torch.rand(heads,img_size,att_rank)-1)
                K = np.sqrt(1/img_size)*(2*torch.rand(heads,att_rank,img_size)-1)

            # elif init[:2] == 'mi':
            #     a, b, c = init[2:].split('_')
            #     img_size = int(a)
            #     att_rank = int(b)
            #     ff = int(c)
            #     weight = torch.zeros((heads,img_size**2,img_size**2))
            #     k = torch.randint(0,ff**2,(heads,))
            #     for i in range(heads):
            #         m = (k[i]//ff)-(ff//2)
            #         n = (k[i]%ff)-(ff//2)
            #         tmp_weight = torch.zeros((img_size,img_size))
            #         for j in range(0-min(0,n),img_size-max(0,n)):
            #             tmp_weight[j,j+n] = 1
            #         for j in range(0-min(0,m),img_size-max(0,m)):
            #             weight[i,j*img_size:(j+1)*img_size,(j+m)*img_size:(j+m+1)*img_size] = tmp_weight
            #     W = 0.7*np.sqrt(1/img_size**2)*(2*torch.rand(heads,img_size**2,img_size**2)-1)+0.7*weight
            #     U,s,V = torch.linalg.svd(W)
            #     s_2 = torch.sqrt(s)
            #     Q = torch.matmul(U[:,:,:att_rank], torch.diag_embed(s_2)[:,:att_rank,:att_rank])
            #     K = torch.matmul(torch.diag_embed(s_2)[:,:att_rank,:att_rank], V[:,:att_rank,:])

            
            elif init[:7] == 'mimetic':
                a, b = init[7:].split('_')
                img_size = int(a)
                att_rank = int(b)
                W = 0.7*np.sqrt(1/img_size)*(2*torch.rand(heads,img_size,img_size)-1)+0.7*torch.eye(img_size).unsqueeze(0).repeat(heads,1,1)
                U,s,V = torch.linalg.svd(W)
                s_2 = torch.sqrt(s)
                Q = torch.matmul(U[:,:,:att_rank], torch.diag_embed(s_2)[:,:att_rank,:att_rank])
                K = torch.matmul(torch.diag_embed(s_2)[:,:att_rank,:att_rank], V[:,:att_rank,:])
            if self.spatial_pe|self.spatial_x:
                print('use linear format')
                self.to_qk = nn.Linear(dim, inner_dim*2, bias = False)
                self.to_qk.weight.data[:512,:] = rearrange(Q, 'h n d -> n (h d)').T
                self.to_qk.weight.data[512:,:] = rearrange(K, 'h d n -> n (h d)').T
            else:
                print('use Q K format')
                self.Q = nn.Parameter(Q,requires_grad=trainable)
                self.K = nn.Parameter(K,requires_grad=trainable)
            
        # use v or just use x
        self.use_value = use_value
        if use_value:
            self.to_v = nn.Linear(dim, inner_dim, bias = False)
        
        # use output layer or not
        self.out_layer = out_layer
        if self.out_layer:
            self.to_out = nn.Linear(inner_dim, dim, bias = False)
        


    def forward(self, x):
        x = self.norm(x)

        # use v or just use x
        if self.use_value:
            v = self.to_v(x)
        else:
            v = x

        # q&k format
        if self.spatial_pe|self.spatial_x:
            # input to q&v
            device = x.device
            if self.spatial_pe&self.spatial_x:
                x = self.alpha*x + (1-self.alpha)*self.pos_embedding.to(device, dtype=x.dtype)
            elif self.spatial_pe:
                x = 0*x + self.pos_embedding.to(device, dtype=x.dtype)
            qk = self.to_qk(x).chunk(2, dim = -1)
            q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qk)
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale            
        else:
            dots = torch.matmul(self.Q, self.K) * self.scale
        attn = self.attend(dots)

        out = torch.matmul(attn, rearrange(v, 'b n (h d) -> b h n d', h = self.heads))
        out = rearrange(out, 'b h n d -> b n (h d)')
        if self.out_layer:
            return self.to_out(out)
        else:
            return out



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, use_value=True, spatial_pe=None, spatial_x = True, init = 'none', alpha=1.0, trainable=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads, dim_head, use_value, spatial_pe, spatial_x, init, alpha, trainable),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, 
                 input_pe = True, pe_choice='sin', use_value = True, spatial_pe = False, spatial_x = True, init = 'none', alpha=0.5, trainable=False):
        super().__init__()

        self.input_pe = input_pe
        self.use_value = use_value
        self.alpha = alpha
        if input_pe: alpha_inside = 1.0
        else: alpha_inside = alpha

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        if pe_choice == 'sin':
            self.pos_embedding = posemb_sincos_2d(
                h = image_height // patch_height,
                w = image_width // patch_width,
                dim = dim,
            )
        elif pe_choice == 'identity':
            # self.pos_embedding = torch.eye(64).repeat(1,8).type(torch.float32)
            s = (image_height // patch_height)*(image_width // patch_width)
            self.pos_embedding = torch.cat([torch.eye(s),torch.zeros(s,dim-s)],dim=-1).type(torch.float32)
        if spatial_pe:
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, use_value, self.pos_embedding, spatial_x, init, alpha_inside, trainable)
        else: # change dim_heads here
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, use_value, None, spatial_x, init, alpha_inside, trainable)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        if self.input_pe:
            x = self.alpha*x + (1-self.alpha)*self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)