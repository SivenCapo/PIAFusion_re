import torch
from torch import nn
from models.common import reflect_conv
from einops import rearrange
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

def CMDAF(vi_feature, ir_feature):
    sigmoid = nn.Sigmoid()
    gap = nn.AdaptiveAvgPool2d(1)


    batch_size, channels, _, _ = vi_feature.size()

    sub_vi_ir = vi_feature - ir_feature
    vi_ir_div = sub_vi_ir * sigmoid(gap(sub_vi_ir))

    sub_ir_vi = ir_feature - vi_feature
    ir_vi_div = sub_ir_vi * sigmoid(gap(sub_ir_vi))

    # 特征加上各自的带有简易通道注意力机制的互补特征
    vi_feature += ir_vi_div
    ir_feature += vi_ir_div

    return vi_feature, ir_feature

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)



def Fusion(vi_out, ir_out):
    return torch.cat([vi_out, ir_out], dim=1)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias




class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expand_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expand_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expand_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expand_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CoAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CoAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.query = nn.Sequential(*[nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
                                     nn.Conv2d(
                                         dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
                                     ])
        self.key = nn.Sequential(*[nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
                                   nn.Conv2d(dim, dim, kernel_size=3, stride=1,
                                             padding=1, groups=dim, bias=bias)
                                   ])
        self.value = nn.Sequential(*[nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
                                     nn.Conv2d(
                                         dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
                                     ])
        # self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        # self.qkv_dwconv = nn.Conv2d(
        #     dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x_feat, y_feat):
        b, c, h, w = x_feat.shape

        q = self.query(x_feat)
        k = self.key(y_feat)
        v = self.value(y_feat)

        # qkv = self.qkv_dwconv(self.qkv(x_feat))
        # q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class CoAttTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expand_factor, bias, LayerNorm_type):
        super(CoAttTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.coatt = CoAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.att   = Attention(dim, num_heads, bias)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expand_factor, bias)

    def forward(self, x, y):
        x = x + self.coatt(self.norm1(x), self.norm1(y))
        x = x + self.att(self.norm2(x))
        x = x + self.ffn(self.norm3(x))
        return x


class CrossModalityBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expand_factor, bias, LayerNorm_type):
        super(CrossModalityBlock, self).__init__()

        self.vis_coattn = CoAttTransformerBlock(dim=dim, num_heads=num_heads, ffn_expand_factor=ffn_expand_factor,
                                    bias=bias, LayerNorm_type=LayerNorm_type)
        self.ir_coattn  = CoAttTransformerBlock(dim=dim, num_heads=num_heads, ffn_expand_factor=ffn_expand_factor,
                                    bias=bias, LayerNorm_type=LayerNorm_type)

    def forward(self, input_feat):
        vis_feat, ir_feat = input_feat[0], input_feat[1]
        vis_feat_mid = self.vis_coattn(vis_feat, ir_feat)
        ir_feat_mid  = self.ir_coattn(ir_feat, vis_feat)
        return [vis_feat_mid, ir_feat_mid]


class Encoder(nn.Module):
    def __init__(self,
                 dim=16,
                 num_blocks=[2, 2, 2],
                 heads=[8, 8, 8],
                 ffn_expand_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',):
        super(Encoder, self).__init__()
        self.vi_conv1 = nn.Conv2d(in_channels=1, kernel_size=1, out_channels=16, stride=1, padding=0)
        self.ir_conv1 = nn.Conv2d(in_channels=1, kernel_size=1, out_channels=16, stride=1, padding=0)

        self.vi_conv2 = reflect_conv(in_channels=16, kernel_size=3, out_channels=16, stride=1, pad=1)
        self.ir_conv2 = reflect_conv(in_channels=16, kernel_size=3, out_channels=16, stride=1, pad=1)

        self.vi_conv3 = reflect_conv(in_channels=16, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.ir_conv3 = reflect_conv(in_channels=16, kernel_size=3, out_channels=32, stride=1, pad=1)

        #self.vis_att_layer = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expand_factor=ffn_expand_factor,
         #                           bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[1])]) 
        #self.ir_att_layer  = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expand_factor=ffn_expand_factor,
          #                          bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[1])])

        self.coatt_layer1 = nn.Sequential(*[CrossModalityBlock(dim=16, num_heads=heads[1], ffn_expand_factor=ffn_expand_factor,
                                    bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[2])]) 
        self.coatt_layer2 = nn.Sequential(*[CrossModalityBlock(dim=32, num_heads=heads[1], ffn_expand_factor=ffn_expand_factor,
                                    bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[2])]) 
        #self.coatt_layer3 = nn.Sequential(*[CrossModalityBlock(dim=64, num_heads=heads[1], ffn_expand_factor=ffn_expand_factor,
         #                           bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[2])]) 

        self.vi_conv4 = reflect_conv(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.ir_conv4 = reflect_conv(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)

        self.vi_conv5 = reflect_conv(in_channels=64, kernel_size=3, out_channels=128, stride=1, pad=1)
        self.ir_conv5 = reflect_conv(in_channels=64, kernel_size=3, out_channels=128, stride=1, pad=1)

    def forward(self, y_vi_image, ir_image):
        activate = nn.LeakyReLU()
        vi_out = activate(self.vi_conv1(y_vi_image))
        ir_out = activate(self.ir_conv1(ir_image))

        vi_out, ir_out = activate(self.vi_conv2(vi_out)), activate(self.ir_conv2(ir_out))
        vi_out, ir_out = CMDAF(vi_out, ir_out )
        vis_coatt_feat, ir_coatt_feat = self.coatt_layer1([vi_out, ir_out])
        vi_out = vi_out + vis_coatt_feat
        ir_out = ir_out + ir_coatt_feat

        vi_out, ir_out = activate(self.vi_conv3(vi_out)), activate(self.ir_conv3(ir_out))
        vi_out, ir_out = CMDAF(vi_out, ir_out )
        vis_coatt_feat, ir_coatt_feat = self.coatt_layer2([vi_out, ir_out])
        vi_out = vi_out + vis_coatt_feat
        ir_out = ir_out + ir_coatt_feat
        
        vi_out, ir_out = activate(self.vi_conv4(vi_out)), activate(self.ir_conv4(ir_out))
        vi_out, ir_out = CMDAF(vi_out, ir_out )
        #vis_coatt_feat, ir_coatt_feat = self.coatt_layer3([vi_out, ir_out])
        #vi_out = vi_out + vis_coatt_feat
        #ir_out = ir_out + ir_coatt_feat

        vi_out, ir_out = activate(self.vi_conv5(vi_out)), activate(self.ir_conv5(ir_out))
        return vi_out, ir_out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = reflect_conv(in_channels=256, kernel_size=3, out_channels=256, stride=1, pad=1)
        self.conv2 = reflect_conv(in_channels=256, kernel_size=3, out_channels=128, stride=1, pad=1)
        self.conv3 = reflect_conv(in_channels=128, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.conv4 = reflect_conv(in_channels=64, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.conv5 = nn.Conv2d(in_channels=32, kernel_size=1, out_channels=1, stride=1, padding=0)

    def forward(self, x):
        activate = nn.LeakyReLU()
        x = activate(self.conv1(x))
        x = activate(self.conv2(x))
        x = activate(self.conv3(x))
        x = activate(self.conv4(x))
        x = nn.Tanh()(self.conv5(x)) / 2 + 0.5  # 将范围从[-1,1]转换为[0,1]
        return x


class PIAFusion(nn.Module):
    def __init__(self):
        super(PIAFusion, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, y_vi_image, ir_image):
        vi_encoder_out, ir_encoder_out = self.encoder(y_vi_image, ir_image)
        encoder_out = Fusion(vi_encoder_out, ir_encoder_out)
        fused_image = self.decoder(encoder_out)
        return fused_image
