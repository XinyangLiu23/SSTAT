import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange


class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvAct(nn.Module):

    def __init__(self, in_channel, out_channel, kernel, stride, padding, dilation=1, bias=False, act=True):
        super(ConvAct, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        #x = self.bn(x)
        x = self.act(x)
        return x


class PyramidPoolingModule(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(PyramidPoolingModule, self).__init__()
        inter_channels = in_channels //4
        self.ca1 = ConvAct(in_channels, inter_channels, 1, 1, 0)
        self.ca2 = ConvAct(in_channels, inter_channels, 1, 1, 0)
        self.ca3 = ConvAct(in_channels, inter_channels, 1, 1, 0)
        self.ca4 = ConvAct(in_channels, inter_channels, 1, 1, 0)
        self.out  = ConvAct(in_channels * 2, out_channels, 1, 1, 0)

    def pool(self, x, size):
        return nn.AdaptiveAvgPool2d(size)(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode="bilinear", align_corners=True)

    def forward(self, x):
        size = x.shape[2:]
        f1 = self.upsample(self.ca1(self.pool(x, 1)), size)
        f2 = self.upsample(self.ca2(self.pool(x, 2)), size)
        f3 = self.upsample(self.ca3(self.pool(x, 3)), size)
        f4 = self.upsample(self.ca4(self.pool(x, 6)), size)
        f = torch.cat([x, f1, f2, f3, f4], dim=1)
        return self.out(f)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class MS_MSA(nn.Module):
    def __init__(self, dim, dim_head, heads,):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class MSAB(nn.Module):
    def __init__(self, dim, dim_head, heads, num_blocks,):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim)),
            ]))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class SpectralChannelTransformer(nn.Module):
    def __init__(self, channels, dim_head=16, heads=3, depth=2):
        super().__init__()
        self.msab = MSAB(dim=channels, dim_head=dim_head, heads=heads, num_blocks=depth)

    def forward(self, x):
        # x: [B, C, H, W]
        out = self.msab(x)
        return out


class SSTAT_decoder(nn.Module):
    def __init__(self, in_channels, out_channels, decoder_params):
        super().__init__()
        c1_in, c2_in, c3_in, c4_in = in_channels
        embed_dim = decoder_params['embed_dim']

        self.ppm = PyramidPoolingModule(512, c4_in)
        self.linear_c4 = MLP(input_dim=c4_in, embed_dim=embed_dim)
        self.linear_c3 = MLP(input_dim=c3_in, embed_dim=embed_dim)
        self.linear_c2 = MLP(input_dim=c2_in, embed_dim=embed_dim)
        self.linear_c1 = MLP(input_dim=c1_in, embed_dim=embed_dim)

        # fuse & pred
        self.linear_fuse = ConvModule(embed_dim * 4, embed_dim, kernel_size=1, norm=False, activation=True)
        self.linear_pred = nn.Conv2d(embed_dim, out_channels, kernel_size=1)
        self.fusion_alpha = nn.Parameter(torch.zeros(out_channels))
        self.ssa = SpectralChannelTransformer(channels=out_channels,
                                                dim_head=24,
                                                heads=2,
                                                depth=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def _expand_rgb_to_hsi(self, rgb, H, W):
        B, C, H0, W0 = rgb.shape
        assert C == 3, f"expect 3-channel RGB, got {C}"
        assert H0 == H and W0 == W

        b = rgb[:, 0:1, :, :]
        g = rgb[:, 1:2, :, :]
        r = rgb[:, 2:3, :, :]

        b_rep = b.repeat(1, 8, 1, 1)  # [B,16,H,W]
        g_rep = g.repeat(1, 7, 1, 1)  # [B,16,H,W]
        r_rep = r.repeat(1, 33, 1, 1)  # [B,16,H,W]

        rgb_expanded = torch.cat([b_rep, g_rep, r_rep], dim=1)  # [B,48,H,W]
        return rgb_expanded

    def forward(self, inputs, output_size, rgb):
        c1, c2, c3, c4 = inputs
        n, _, h, w = c4.shape
        H_out, W_out = output_size

        c4 = self.ppm(c4)
        _c4 = F.interpolate(self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, h, w),
                            size=c1.shape[2:], mode='bilinear', align_corners=True)
        _c3 = F.interpolate(self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3]),
                            size=c1.shape[2:], mode='bilinear', align_corners=True)
        _c2 = F.interpolate(self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3]),
                            size=c1.shape[2:], mode='bilinear', align_corners=True)
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        # fuse
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        _c = self.linear_pred(_c)
        x = self.conv_relu(_c) #FLOPs
        x = F.interpolate(x, output_size, mode='bilinear', align_corners=False)

        rgb_expanded = self._expand_rgb_to_hsi(rgb, H_out, W_out)  # [B,48,H,W]

        alpha = torch.sigmoid(self.fusion_alpha).view(1, -1, 1, 1)  # [1,48,1,1]
        fused = alpha * x + (1.0 - alpha) * rgb_expanded  # [B,48,H,W]

        x = fused + self.ssa(fused)
        x = self.refine(x)
        out = torch.clamp(F.relu(x), max=12.0)

        return out

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 norm=True, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=not norm)
        self.norm = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        #x = self.norm(x)
        x = self.act(x)
        return x