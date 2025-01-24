import copy
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict
from customized_linear import CustomizedLinear
from einops import rearrange


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AdaptiveDropout(nn.Module):
    def __init__(self, dropout_rate, adapt_rate):
        super(AdaptiveDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.adapt_rate = adapt_rate
    def forward(self, x):
        if self.training:
            mask_adapt = np.random.binomial(1, self.dropout_rate, size=x.shape[1])
            mask_adapt = torch.from_numpy(mask_adapt).float().to(x.device)
            x = x * mask_adapt / (1 - self.dropout_rate)
            self.dropout_rate -= self.adapt_rate * (torch.mean(mask_adapt) - self.dropout_rate)
            self.dropout_rate = self.dropout_rate.item()
        return x


class FeatureEmbed(nn.Module):
    def __init__(self, num_genes, mask, embed_dim=192, fe_bias=True, norm_layer=None):
        super().__init__()
        self.num_genes = num_genes
        self.num_patches = mask.shape[1]
        self.embed_dim = embed_dim
        mask = np.repeat(mask,embed_dim,axis=1)
        self.mask = mask
        self.fe = CustomizedLinear(self.mask)  # Cell Embedding layer
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x):
        num_cells = x.shape[0]
        x = rearrange(self.fe(x), 'h (w c) -> h c w ', c=self.num_patches)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # token-->Q, K, V
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # q@k = q.matmul(k)
        attn = attn.softmax(dim=-1)
        weights = attn
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, weights


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


'''
class FourierKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, gridsize=4, addbias=True):
        super(FourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim
        self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) /
                                          (np.sqrt(inputdim) * np.sqrt(self.gridsize)))
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))
    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)
        k = torch.reshape(torch.arange(1, self.gridsize + 1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
        cs = [c, s]
        y = torch.einsum('dxik,dyik->xy', torch.cat(cs, dim=0), self.fouriercoeffs)
        if self.addbias:
            y += self.bias
        y = y.view(outshape)
        return y
'''


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
    def forward(self, x):
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        hhh, weights = self.attn(self.norm1(x))
        x = x + self.drop_path(hhh)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, weights


def get_weight(att_mat):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    att_mat = torch.stack(att_mat).squeeze(1)
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=2)
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(3))
    aug_att_mat = att_mat.to(device) + residual_att.to(device)
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    # Recursively multiply the weight matrices.
    joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
    joint_attentions[0] = aug_att_mat[0]
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    v = v[:,0,1:]
    return v


class Transformer(nn.Module):
    def __init__(self, num_classes, num_genes, mask_1, mask_2, mask_3, mask_4, fe_bias=True,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 qkv_bias=True, qk_scale=None, representation_size=None, distilled=False,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                 embed_layer=FeatureEmbed, norm_layer=None, act_layer=None):
        """
        Args:
            num_classes (int): number of classes for classification head
            num_genes (int): number of feature of input(expData)
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): feature embed layer
            norm_layer: (nn.Module): normalization layer
        """
        super(Transformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.feature_embed_1 = embed_layer(num_genes, mask=mask_1, embed_dim=embed_dim, fe_bias=fe_bias)
        self.feature_embed_2 = embed_layer(num_genes, mask=mask_2, embed_dim=embed_dim, fe_bias=fe_bias)
        self.feature_embed_3 = embed_layer(num_genes, mask=mask_3, embed_dim=embed_dim, fe_bias=fe_bias)
        self.feature_embed_4 = embed_layer(num_genes, mask=mask_4, embed_dim=embed_dim, fe_bias=fe_bias)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        # self.blocks = nn.Sequential(*[
        #     Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #           drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
        #           norm_layer=norm_layer, act_layer=act_layer)
        #     for i in range(depth)
        # ])
        self.blocks = nn.ModuleList()
        for i in range(depth):
            layer = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                          norm_layer=norm_layer, act_layer=act_layer)
            self.blocks.append(copy.deepcopy(layer))
        self.norm = norm_layer(embed_dim)
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()
        # Classifier head(s)
        self.head_1 = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_2 = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_3 = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_4 = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head = nn.Linear(self.num_features*4, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        # Weight init
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)
        self.dropout = AdaptiveDropout(0.1, 0.01)

    def forward_features_1(self, x):
        x = self.feature_embed_1(x)
        # abstracts the information during the following network layers and is used to predict the cell type.
        # the attention scores between CLS and pathway tokens mean the importance of the latter to the classification and identification the cell type.
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        attn_weights = []
        tem = x
        for layer_block in self.blocks:
            tem, weights = layer_block(tem)
            attn_weights.append(weights)
        x = self.norm(tem)
        attn_weights = get_weight(attn_weights)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]), attn_weights
        else:
            return x[:, 0], x[:, 1], attn_weights

    def forward_features_2(self, x):
        x = self.feature_embed_2(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        attn_weights = []
        tem = x
        for layer_block in self.blocks:
            tem, weights = layer_block(tem)
            attn_weights.append(weights)
        x = self.norm(tem)
        attn_weights = get_weight(attn_weights)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]), attn_weights
        else:
            return x[:, 0], x[:, 1], attn_weights

    def forward_features_3(self, x):
        x = self.feature_embed_3(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        attn_weights = []
        tem = x
        for layer_block in self.blocks:
            tem, weights = layer_block(tem)
            attn_weights.append(weights)
        x = self.norm(tem)
        attn_weights = get_weight(attn_weights)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]), attn_weights
        else:
            return x[:, 0], x[:, 1], attn_weights

    def forward_features_4(self, x):
        x = self.feature_embed_4(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        attn_weights = []
        tem = x
        for layer_block in self.blocks:
            tem, weights = layer_block(tem)
            attn_weights.append(weights)
        x = self.norm(tem)
        attn_weights = get_weight(attn_weights)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]), attn_weights
        else:
            return x[:, 0], x[:, 1], attn_weights
    def forward(self, x):
        # Cell Embedding layer & Multi-head Self-attention layer
        cls1, attn_weights_1 = self.forward_features_1(x)
        cls2, attn_weights_2 = self.forward_features_2(x)
        cls3, attn_weights_3 = self.forward_features_3(x)
        cls4, attn_weights_4 = self.forward_features_4(x)
        latent = torch.cat((cls1, cls2, cls3, cls4), 1)
        # Single-view classifier
        output1 = self.head_1(cls1)
        output2 = self.head_2(cls2)
        output3 = self.head_3(cls3)
        output4 = self.head_4(cls4)
        # Cell-Type Classifier
        if self.head_dist is not None:
            latent, latent_dist = self.head(latent[0]), self.head_dist(latent[1])
            if self.training and not torch.jit.is_scripting():
                return latent, latent_dist
            else:
                return (latent+latent_dist) / 2
        else:
            pre = self.head(latent)
        return latent, pre, attn_weights_2, (output1, output2, output3, output4, cls1, cls2, cls3, cls4)


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def scTrans_model(num_classes, num_genes, mask_1, mask_2, mask_3, mask_4, embed_dim=60, depth=2, num_heads=4, has_logits: bool = True):  # # embed_dim
    model = Transformer(num_classes=num_classes,
                        num_genes=num_genes,
                        mask_1=mask_1, mask_2=mask_2, mask_3=mask_3, mask_4=mask_4,
                        embed_dim=embed_dim,
                        depth=depth,
                        num_heads=num_heads,
                        drop_ratio=0.5, attn_drop_ratio=0.5, drop_path_ratio=0.5,
                        representation_size=embed_dim if has_logits else None)
    return model
