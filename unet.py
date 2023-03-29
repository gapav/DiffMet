import torch
import torch
from torch import nn
import math
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        self.emb_conv = nn.Conv2d(512, out_ch, kernel_size=1)

        self.conv_chs_add = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv_middle2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.conv_final3 = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.GELU = nn.GELU()
        self.ReLU = nn.ReLU()
        # same normalization as MCVD:
        self.GN = nn.GroupNorm(1, out_ch)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x, t, emb_input):
        # get cond_embedding
        _btch, _ch, img_size1, img_size2 = x.shape
        # upscale emb input to match x:
        upscaled_emb = F.interpolate(
            emb_input, size=(img_size1, img_size2), mode="nearest"
        )

        convd_embbedding = self.emb_conv(upscaled_emb)
        convd_embbedding = self.ReLU(convd_embbedding)

        # get time emb and activate
        t = self.time_mlp(t)
        time_emb = self.ReLU(t)
        time_emb = time_emb[:, :, None, None]

        x = self.conv_chs_add(x)

        # add conditional embedding
        x = x + convd_embbedding + time_emb
        x = self.GN(x)
        x = self.GELU(x)
        x = self.conv_middle2(x)
        x = self.GN(x)

        x = self.GN(x)
        x = self.GELU(x)

        x = self.conv_final3(x)
        x = self.dropout(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        self.emb_conv = nn.Conv2d(512, out_ch, kernel_size=1)

        self.conv_chs_rm = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
        self.conv_middle2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.conv_final3 = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        self.GELU = nn.GELU()
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        # nsame normalization as MCVD:
        self.GN = nn.GroupNorm(1, out_ch)

    def forward(self, x, t, emb_input):
        # get cond_embedding
        _btch, _ch, img_size1, img_size2 = x.shape
        # upscale emb input to match x:
        upscaled_emb = F.interpolate(
            emb_input, size=(img_size1, img_size2), mode="nearest"
        )

        convd_embbedding = self.emb_conv(upscaled_emb)
        convd_embbedding = self.ReLU(convd_embbedding)

        # get time emb and activate
        t = self.time_mlp(t)
        time_emb = self.ReLU(t)
        time_emb = time_emb[:, :, None, None]

        # add conditional embedding
        x = self.conv_chs_rm(x)

        x = x + convd_embbedding
        x = self.GN(x)
        x = x + time_emb
        x = self.GELU(x)
        x = self.conv_middle2(x)
        x = x + convd_embbedding
        x = self.GN(x)
        x = x + time_emb

        x = self.GN(x)
        x = self.GELU(x)

        x = self.conv_final3(x)

        x = self.dropout(x)
        return x


class UNet_embedding(nn.Module):
    def __init__(self, rgb_grayscale, num_cond_frames, device):
        self.rgb_grayscale = rgb_grayscale
        self.num_cond_frames = num_cond_frames
        self.device = device
        self.emb_out = []
        self.time_emb_dim = 256
        self.image_channels = 1
        self.image_channels_out = 1
        self.res = []

        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            # https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
            SinusoidalPositionEmbeddings(self.time_emb_dim),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.ReLU(),
        )
        self.init_conv = nn.Conv2d(self.image_channels, 64, 3, padding=1)
        # Downsample
        self.down1 = Encoder(in_ch=64, out_ch=128, time_emb_dim=self.time_emb_dim)
        self.down2 = Encoder(in_ch=128, out_ch=256, time_emb_dim=self.time_emb_dim)
        self.down3 = Encoder(in_ch=256, out_ch=512, time_emb_dim=self.time_emb_dim)
        self.down4 = Encoder(in_ch=512, out_ch=1024, time_emb_dim=self.time_emb_dim)

        # Upsample
        self.up1 = Decoder(in_ch=1024, out_ch=512, time_emb_dim=self.time_emb_dim)
        self.up2 = Decoder(in_ch=512, out_ch=256, time_emb_dim=self.time_emb_dim)
        self.up3 = Decoder(in_ch=256, out_ch=128, time_emb_dim=self.time_emb_dim)
        self.up4 = Decoder(in_ch=128, out_ch=64, time_emb_dim=self.time_emb_dim)

        self.out_conv = nn.Conv2d(64, 1, self.image_channels_out)

    def forward(self, x, timestep, condition, cond_emb_model):
        # cond
        # Embedd time
        t = self.time_mlp(timestep)
        x = x.to(self.device)
        # access layer before avgpool
        _embedding_layer = cond_emb_model._modules.get("layer4")
        # extract output
        _ = _embedding_layer.register_forward_hook(self.hook)

        # run data thorugh resnet to get embedding from hook:
        cond_emb_model(condition).detach()

        # Initial conv

        # DOWNSAMPLE:

        self.res = []

        x = self.init_conv(x)
        emb = self.emb_out[0]
        emb = emb.to(self.device)

        x = self.down1(x, t, emb)
        self.res.append(x)

        x = self.down2(x, t, emb)
        self.res.append(x)

        x = self.down3(x, t, emb)
        self.res.append(x)

        x = self.down4(x, t, emb)
        self.res.append(x)

        # UP SAMPLE:
        residual_x = self.res.pop()
        # residual_x = residual_x.to(self.device)
        x = torch.cat((x, residual_x), dim=1)
        x = self.up1(x, t, emb)

        residual_x = self.res.pop()
        # residual_x = residual_x.to(self.device)
        # concat x with self.res channels
        x = torch.cat((x, residual_x), dim=1)
        x = self.up2(x, t, emb)

        residual_x = self.res.pop()
        # residual_x = residual_x.to(self.device)
        x = torch.cat((x, residual_x), dim=1)
        x = self.up3(x, t, emb)

        residual_x = self.res.pop()
        # residual_x = residual_x.to(self.device)
        x = torch.cat((x, residual_x), dim=1)
        x = self.up4(x, t, emb)

        x = self.out_conv(x)

        return x

    def hook(self, module, input, output):
        """
        append output from  layer4 from resnet18
        """
        # clear embedding list from previous forward call:
        self.emb_out = []
        # append embedding:
        self.emb_out.append(output.detach())


class SinusoidalPositionEmbeddings(nn.Module):

    """ 
    https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


if __name__ == "__main__":
    sample = torch.rand(1, 5, 64, 64)  # batch, channels, H,W

