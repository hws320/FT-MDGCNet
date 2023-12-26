# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import numpy as np
import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select, act_layer
from .torch_edge import DenseDilatedKnnGraph
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        return self.nn(x)


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GraphSAGE(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv2d(nn.Module):
    """
    GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)


class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)     ####(B,C,H/r,W/r)
            y = y.reshape(B, C, -1, 1).contiguous()   ###(B,C,HW/(r*r))         
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x, y, relative_pos)    ####relative_pos:(1,HW,HW//(r*r))
        x = super(DyGraphConv2d, self).forward(x, edge_index, y)
        return x.reshape(B, -1, H, W).contiguous()


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196,H=56,W=56, drop_path=0.0, relative_pos=False):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, r)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            print('using relative_pos')
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                H,W))).unsqueeze(0).unsqueeze(1)     ####(1,1,HW,HW)
            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)    ####interpolate上下采样函数，输出大小为参数size
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)   ####(1,HW,HW//(r*r))

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)    #####(1,HW,HW//(r*r))
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x


class Grapher_conv(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196,H=56,W=56, drop_path=0.0, relative_pos=False):
        super(Grapher_conv, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, r)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            print('using relative_pos')
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                H,W))).unsqueeze(0).unsqueeze(1)     ####(1,1,HW,HW)
            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)    ####interpolate上下采样函数，输出大小为参数size
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)   ####(1,HW,HW//(r*r))

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x):
        _tmp = x
        # x = self.fc1(x)
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)    #####(1,HW,HW//(r*r))
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        x = self.drop_path(x)
        return x


# class Grapher_3d(nn.Module):
#     """
#     Grapher module with graph convolution and fc layers
#     """
#     def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
#                  bias=True,  stochastic=False, epsilon=0.0, r=1, n=196,H=56,W=56, drop_path=0.0, relative_pos=False):
#         super(Grapher_3d, self).__init__()
#         self.channels = in_channels
#         self.n = n
#         self.r = r
#         # self.fc1 = nn.Sequential(
#         #     nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
#         #     nn.BatchNorm2d(in_channels),
#         # )
#         self.graph_conv_c = Grapher_Conv(in_channels,kernel_size,dilation,conv,act,norm,bias,stochastic,epsilon,r,n,H,W,relative_pos)     ####(B,C,H,W)
#         self.graph_conv_h = Grapher_Conv(H,kernel_size,dilation,conv,act,norm,bias,stochastic,epsilon,1,in_channels*W,in_channels,W,relative_pos)  ###(B,H,C,W)
#         self.graph_conv_w = Grapher_Conv(W,kernel_size,dilation,conv,act,norm,bias,stochastic,epsilon,1,in_channels*H,H,in_channels,relative_pos)  ###(B,W,H,C)
#         self.fc2 = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
#             nn.BatchNorm2d(in_channels),
#         )
#         self.reweight = Mlp(in_channels, in_channels // 4, in_channels * 3)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         _tmp = x
#         B, C, H, W = x.shape
#         # x = self.fc1(x)


#         x_c = self.graph_conv_c(x)     ####(B,C,H,W)

#         x_h = x.transpose(1,2).contiguous()  ###(B,H,C,W)
#         x_h = self.graph_conv_h(x_h).transpose(1,2).contiguous()   ###(B,C,H,W)

#         x_w = x.transpose(1,3).contiguous()   ###(B,W,H,C)
#         x_w = self.graph_conv_w(x_w).transpose(1,3).contiguous()  ###(B,C,H,W)

#         a = (x_h + x_w + x_c).flatten(2).mean(2).unsqueeze(2).unsqueeze(2)   ###(B,C,1,1)
#         a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(3).unsqueeze(3)   ####(3,B,C,1,1)

#         x = x_h * a[0] + x_w * a[1] + x_c * a[2]
        
#         x = self.fc2(x)
#         x = self.drop_path(x) + _tmp
#         return x



class Grapher_3d_1(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196,H=56,W=56, drop_path=0.0, relative_pos=False):
        super(Grapher_3d_1, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv_c = Grapher_conv(in_channels,kernel_size,dilation,conv,act,norm,bias,stochastic,epsilon,r,n,H,W,drop_path,relative_pos)     ####(B,C,H,W)
        self.graph_conv_h = Grapher_conv(H,kernel_size,dilation,conv,act,norm,bias,stochastic,epsilon,1,in_channels*W,in_channels,W,drop_path,relative_pos)  ###(B,H,C,W)
        self.graph_conv_w = Grapher_conv(W,kernel_size,dilation,conv,act,norm,bias,stochastic,epsilon,1,in_channels*H,H,in_channels,drop_path,relative_pos)  ###(B,W,H,C)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.reweight = Mlp(in_channels, in_channels // 4, in_channels * 3)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        _tmp = x
        B, C, H, W = x.shape
        x = self.fc1(x)


        x_c = self.graph_conv_c(x)     ####(B,C,H,W)

        x_h = x.transpose(1,2).contiguous()  ###(B,H,C,W)
        x_h = self.graph_conv_h(x_h).transpose(1,2).contiguous()   ###(B,C,H,W)

        x_w = x.transpose(1,3).contiguous()   ###(B,W,H,C)
        x_w = self.graph_conv_w(x_w).transpose(1,3).contiguous()  ###(B,C,H,W)

        a = (x_h + x_w + x_c).flatten(2).mean(2).unsqueeze(2).unsqueeze(2)   ###(B,C,1,1)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(3).unsqueeze(3)   ####(3,B,C,1,1)

        x = x_h * a[0] + x_w * a[1] + x_c * a[2]
        
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x


class Grapher_3d_2(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196,H=56,W=56, drop_path=0.0, relative_pos=False):
        super(Grapher_3d_2, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv_c = Grapher_conv(in_channels,kernel_size,dilation,conv,act,norm,bias,stochastic,epsilon,r,n,H,W,drop_path,relative_pos)     ####(B,C,H,W)
        self.graph_conv_h = Grapher_conv(H*2,kernel_size,dilation,conv,act,norm,bias,stochastic,epsilon,1,in_channels*W//2,in_channels//2,W,drop_path,relative_pos)  ###(B,H,C,W)
        self.graph_conv_w = Grapher_conv(W*2,kernel_size,dilation,conv,act,norm,bias,stochastic,epsilon,1,in_channels*H//2,H,in_channels//2,drop_path,relative_pos)  ###(B,W,H,C)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )##########采用Concat时，上两行的代码中，in_channels要乘3
        self.reweight = Mlp(in_channels, in_channels // 4, in_channels * 3)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        _tmp = x
        B, C, H, W = x.shape
        x = self.fc1(x)


        x_c = self.graph_conv_c(x)     ####(B,C,H,W)

        x_h = x.reshape(B,C//2,H*2,W).transpose(1,2)  ###(B,H,C,W)
        x_h = self.graph_conv_h(x_h).transpose(1,2).reshape(B,C,H,W)   ###(B,C,H,W)

        x_w = x.reshape(B,C//2,H,W*2).transpose(1,3)   ###(B,W,H,C)
        x_w = self.graph_conv_w(x_w).transpose(1,3).reshape(B,C,H,W) ###(B,C,H,W)

        ########加权融合
        a = (x_h + x_w + x_c).flatten(2).mean(2).unsqueeze(2).unsqueeze(2)   ###(B,C,1,1)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(3).unsqueeze(3)   ####(3,B,C,1,1)
        x = x_h * a[0] + x_w * a[1] + x_c * a[2]

        # ######C+H的加权融合
        # a = (x_h + x_c).flatten(2).mean(2).unsqueeze(2).unsqueeze(2)   ###(B,C,1,1)
        # a = self.reweight(a).reshape(B, C, 2).permute(2, 0, 1).softmax(dim=0).unsqueeze(3).unsqueeze(3)   ####(3,B,C,1,1)
        # x = x_h * a[0] + x_c * a[1]

        # ######C+W的加权融合
        # a = (x_w + x_c).flatten(2).mean(2).unsqueeze(2).unsqueeze(2)   ###(B,C,1,1)
        # a = self.reweight(a).reshape(B, C, 2).permute(2, 0, 1).softmax(dim=0).unsqueeze(3).unsqueeze(3)   ####(3,B,C,1,1)
        # x = x_w * a[0] + x_c * a[1]


        
        # x = x_c + x_h + x_w         ####采用Add的方式特征融合


        # x = torch.cat([x_c,x_h,x_w],1)  #####采用Concat的方式特征融合

        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x


class Grapher_3d_3(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196,H=56,W=56, drop_path=0.0, relative_pos=False):
        super(Grapher_3d_3, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv_c = Grapher_conv(in_channels,kernel_size,dilation,conv,act,norm,bias,stochastic,epsilon,r,n,H,W,drop_path,relative_pos)     ####(B,C,H,W)
        self.graph_conv_h = Grapher_conv(H*4,kernel_size,dilation,conv,act,norm,bias,stochastic,epsilon,1,in_channels*W//4,in_channels//4,W,drop_path,relative_pos)  ###(B,H,C,W)
        self.graph_conv_w = Grapher_conv(W*4,kernel_size,dilation,conv,act,norm,bias,stochastic,epsilon,1,in_channels*H//4,H,in_channels//4,drop_path,relative_pos)  ###(B,W,H,C)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.reweight = Mlp(in_channels, in_channels // 4, in_channels * 3)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        _tmp = x
        B, C, H, W = x.shape
        x = self.fc1(x)


        x_c = self.graph_conv_c(x)     ####(B,C,H,W)

        x_h = x.reshape(B,C//4,H*4,W).transpose(1,2)  ###(B,H,C,W)
        x_h = self.graph_conv_h(x_h).transpose(1,2).reshape(B,C,H,W)   ###(B,C,H,W)

        x_w = x.reshape(B,C//4,H,W*4).transpose(1,3)   ###(B,W,H,C)
        x_w = self.graph_conv_w(x_w).transpose(1,3).reshape(B,C,H,W) ###(B,C,H,W)

        a = (x_h + x_w + x_c).flatten(2).mean(2).unsqueeze(2).unsqueeze(2)   ###(B,C,1,1)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(3).unsqueeze(3)   ####(3,B,C,1,1)

        x = x_h * a[0] + x_w * a[1] + x_c * a[2]
        
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x


class Grapher_3d_4(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196,H=56,W=56, drop_path=0.0, relative_pos=False):
        super(Grapher_3d_4, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv_c = Grapher_conv(in_channels,kernel_size,dilation,conv,act,norm,bias,stochastic,epsilon,r,n,H,W,drop_path,relative_pos)     ####(B,C,H,W)
        self.graph_conv_h = Grapher_conv(H*8,kernel_size,dilation,conv,act,norm,bias,stochastic,epsilon,1,in_channels*W//8,in_channels//8,W,drop_path,relative_pos)  ###(B,H,C,W)
        self.graph_conv_w = Grapher_conv(W*8,kernel_size,dilation,conv,act,norm,bias,stochastic,epsilon,1,in_channels*H//8,H,in_channels//8,drop_path,relative_pos)  ###(B,W,H,C)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.reweight = Mlp(in_channels, in_channels // 4, in_channels * 3)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        _tmp = x
        B, C, H, W = x.shape
        x = self.fc1(x)


        x_c = self.graph_conv_c(x)     ####(B,C,H,W)

        x_h = x.reshape(B,C//8,H*8,W).transpose(1,2)  ###(B,H,C,W)
        x_h = self.graph_conv_h(x_h).transpose(1,2).reshape(B,C,H,W)   ###(B,C,H,W)

        x_w = x.reshape(B,C//8,H,W*8).transpose(1,3)   ###(B,W,H,C)
        x_w = self.graph_conv_w(x_w).transpose(1,3).reshape(B,C,H,W) ###(B,C,H,W)

        a = (x_h + x_w + x_c).flatten(2).mean(2).unsqueeze(2).unsqueeze(2)   ###(B,C,1,1)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(3).unsqueeze(3)   ####(3,B,C,1,1)

        x = x_h * a[0] + x_w * a[1] + x_c * a[2]
        
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x