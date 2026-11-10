import torch.nn as nn
from architecture.decoder import SSTAT_decoder
from architecture.encoder import SSTAT_encoder

class generator(nn.Module):
    def __init__(self, input_nc=3, embed_dim=256, pretrained=None, output_nc=48):
        super().__init__()
        self.encoder = SSTAT_encoder(
            img_size=128,
            in_chans=input_nc,
            embed_dims=[64, 128, 320, 512],
            depths=[3, 4, 6, 3],
            num_heads=[1, 2, 4, 8],
            sr_ratios=[8, 4, 2, 1]
        )

        if pretrained:
            self.encoder.init_weights(pretrained)

        #decoder
        in_channels=[64, 128, 320, 512]
        self.decoder = SSTAT_decoder(
            in_channels=in_channels,
            out_channels=output_nc,
            decoder_params=dict(embed_dim=embed_dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        feats = self.encoder.forward_features(x)
        out = self.decoder(feats, output_size=(H, W), rgb=x)
        return out

