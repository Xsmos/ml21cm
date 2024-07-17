# from dataclasses import dataclass
# import h5py
import torch
import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# from datasets import Dataset
import matplotlib.pyplot as plt
import numpy as np
import random
from abc import ABC, abstractmethod
import torch.nn.functional as F
import math
# from PIL import Image
import os
# from torch.utils.tensorboard import SummaryWriter
import copy
# from tqdm.auto import tqdm
# from torchvision import transforms
# from diffusers import UNet2DModel#, UNet3DConditionModel
# from diffusers import DDPMScheduler
# from diffusers.utils import make_image_grid
import datetime
# from pathlib import Path
# from diffusers.optimization import get_cosine_schedule_with_warmup
# from accelerate import notebook_launcher, Accelerator
# from huggingface_hub import create_repo, upload_folder
# from load_h5 import Dataset4h5

class GroupNorm32(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, swish, eps=1e-5):
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps)
        self.swish = swish

    def forward(self, x):
        # print("GroupNorm32, x.dtype =", x.dtype)
        y = super().forward(x.float()).to(x.dtype)
        if self.swish == 1.0:
            y = F.silu(y)
        elif self.swish:
            y = y * F.sigmoid(y * float(self.swish))
        return y

def normalization(channels, swish=0.0):
    """
    Make a standard normalization layer, with an optional swish activation.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    #print (channels)
    return GroupNorm32(num_channels=channels, num_groups=32, swish=swish)

Conv = {
    1: nn.Conv1d,
    2: nn.Conv2d,
    3: nn.Conv3d,
}

AvgPool = {
    1: nn.AvgPool1d,
    2: nn.AvgPool2d,
    3: nn.AvgPool3d
}

class Downsample(nn.Module):
    def __init__(self, channels, use_conv, out_channels=None, dim=2, stride=(2,2)):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        # stride = config.stride
        if use_conv:
            # print("conv")
            self.op = Conv[dim](channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            # print("pool")
            assert channels == self.out_channels
            self.op = AvgPool[dim](kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class Upsample(nn.Module):
    def __init__(self, channels, use_conv, out_channels=None, dim=2, stride=(2,2)):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.use_conv = use_conv
        self.stride = stride
        if self.use_conv:
            self.conv = Conv[dim](self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        # stride = config.stride
        # print(torch.tensor(x.shape[2:]))
        # print(torch.tensor(stride))
        shape = torch.tensor(x.shape[2:]) * torch.tensor(self.stride)
        shape = tuple(shape.detach().numpy())
        # print(shape)
        x = F.interpolate(x, shape, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x

def zero_module(module):
    """
    clean gradient of parameters of the module
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class TimestepBlock(ABC, nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """
        test
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, encoder_out=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, AttentionBlock):
                x = layer(x, encoder_out)
            else:
                x = layer(x)
        return x

class ResBlock(TimestepBlock):
    def __init__(
        self, channels, emb_channels, dropout, out_channels=None, use_conv=False, use_checkpoint=False, use_scale_shift_norm=False, up=False, down=False, dim=2, stride=(2,2),
        ):
        super().__init__()
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.stride = stride

        self.in_layers = nn.Sequential(
            # nn.BatchNorm2d(channels), # normalize to standard gaussian
            normalization(channels, swish=1.0),
            nn.Identity(),
            Conv[dim](channels, self.out_channels, 3, padding=1),
            )

        self.updown = up or down
        if up:
            self.h_updown = Upsample(channels, False, dim=dim, stride=stride)
            self.x_updown = Upsample(channels, False, dim=dim, stride=stride)
        elif down:
            self.h_updown = Downsample(channels, False, dim=dim, stride=stride)
            self.x_updown = Downsample(channels, False, dim=dim, stride=stride)
        else:
            self.h_updown = self.x_updown = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                ),
        )

        self.out_layers = nn.Sequential(
            # nn.BatchNorm2d(self.out_channels),
            normalization(self.out_channels, swish=0.0 if use_scale_shift_norm else 1.0),
            nn.SiLU() if use_scale_shift_norm else nn.Identity(),
            nn.Dropout(p=dropout),
            zero_module(Conv[dim](self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = Conv[dim](channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = Conv[dim](channels, self.out_channels, 1)
        

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_updown(h)
            x = self.x_updown(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        # print("forward, h.dtype =", h.dtype)
        emb_out = self.emb_layers(emb).type(h.dtype)

        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1+scale) + shift
            h = out_rest(h)
        else:
            h += emb_out
            h = self.out_layers(h)
        # print("ResBlock, torch.unique(h).shape =", torch.unique(h).shape)
        return self.skip_connection(x) + h

class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
        # print("QKVAttention, self.n_heads =", self.n_heads)
        
    def forward(self, qkv, encoder_kv=None):
        bs, width, length = qkv.shape
        assert width % (3*self.n_heads) == 0
        ch = width // (3*self.n_heads)

        # print("QKVAttention", bs, self.n_heads, ch, length)
        q, k, v = qkv.reshape(bs*self.n_heads, ch*3, length).split(ch, dim=1)
        if encoder_kv is not None:
            assert encoder_kv.shape[1] == self.n_heads * ch * 2
            ek, ev = encoder_kv.reshape(bs*self.n_heads, ch*2, -1).split(ch, dim=1)
            k = torch.cat([ek,k], dim=-1)
            v = torch.cat([ev,v], dim=-1)

        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q*scale, k*scale)
        # print("forward, weight.dtype =", weight.dtype)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        encoder_channels=None,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0,\
                f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.use_checkpoint = use_checkpoint
        # self.norm = nn.BatchNorm2d(channels)
        self.norm = normalization(channels, swish=0.0)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        
        self.attention = QKVAttention(self.num_heads)

        if encoder_channels is not None:
            self.encoder_kv = nn.Conv1d(encoder_channels, channels * 2, 1)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x, encoder_out=None):
        b, c, *spatial = x.shape
        qkv = self.qkv(self.norm(x).view(b, c, -1))
        if encoder_out is not None:
            encoder_out = self.encoder_kv(encoder_out)
            h = self.attention(qkv, encoder_out)
        else:
            h = self.attention(qkv)
        # print("AttentionBlock, before proj_out, torch.unique(h).shape =", torch.unique(h).shape)
        h = self.proj_out(h)
        # print("AttentionBlock, after proj_out, torch.unique(h).shape =", torch.unique(h).shape)
        return x + h.reshape(b, c, *spatial)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    #print (timesteps.shape)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    #print (timesteps[:, None].float().shape,freqs[None].shape)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class ContextUnet(nn.Module):
    def __init__(
        self,
        n_param=2,
        image_size=64,
        in_channels=1,
        model_channels=128,
        out_channels = 1,
        channel_mult = None,
        num_res_blocks = 2,
        dropout = 0,
        use_checkpoint = False,
        use_scale_shift_norm = False,
        attention_resolutions = (16, 8),
        num_heads = 4,
        num_head_channels = -1,
        num_heads_upsample = -1,
        resblock_updown = False,
        conv_resample = True,
        encoder_channels = None,
        dim = 2,
        stride = (2,2),
        dtype = torch.float32,
        ):
        super().__init__()

        if channel_mult == None:
            if image_size == 512:
                channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
            elif image_size == 256:
                channel_mult = (1, 1, 2, 2, 4, 4)
            elif image_size == 128:
                channel_mult = (1, 1, 2, 3, 4)
            elif image_size == 64:
                channel_mult = (1, 2, 4, 4, 4)#(1, 2, 2, 4)#(1, 2, 8, 8, 8)#(1, 2, 4)#(1, 2, 2, 4)#(0.5,1,2,2,4,4)#(1, 1, 2, 2, 4, 4)#
            elif image_size == 32:
                channel_mult = (1, 2, 2, 4)
            elif image_size == 28:
                channel_mult = (1, 2, 4)#(1, 2, 3, 4)
            else:
                raise ValueError(f"unsupported image size: {image_size}")
        # else:
        #     channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))
        
        attention_ds = []
        for res in attention_resolutions:
            attention_ds.append(image_size // int(res))

        # print("before, ContextUnet, num_heads_upsample =", num_heads_upsample, "num_heads =", num_heads)
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        # print("after, ContextUnet, num_heads_upsample =", num_heads_upsample, "num_heads =", num_heads)

        # self.n_param = n_param
        self.model_channels = model_channels
        # self.use_fp16 = use_fp16
        self.dtype = dtype#torch.float16 if self.use_fp16 else torch.float32

        self.token_embedding = nn.Linear(n_param, model_channels * 4)

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)

        ###################### input_blocks ######################
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(Conv[dim](in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels = int(mult * model_channels),
                        use_checkpoint = use_checkpoint,
                        use_scale_shift_norm = use_scale_shift_norm,
                        dim = dim,
                        stride = stride,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_ds:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads = num_heads,
                            num_head_channels = num_head_channels,
                            encoder_channels = encoder_channels,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            # dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            dim = dim,
                            stride = stride,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, out_channels=out_ch, dim=dim, stride=stride)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch


        ###################### middle_blocks ######################
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                dim = dim,
                stride = stride,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                encoder_channels=encoder_channels,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                dim = dim,
                stride = stride,
            ),
        )
        self._feature_size += ch


        ###################### output_blocks ######################
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        # dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dim = dim,
                        stride = stride,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_ds:
                    # print("ds in attention_resolutions, num_heads=", num_heads_upsample)
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            encoder_channels=encoder_channels,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            # dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            dim = dim,
                            stride = stride,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, out_channels=out_ch, dim=dim, stride=stride)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            # nn.BatchNorm2d(ch),
            normalization(ch, swish=1.0),
            nn.Identity(),
            zero_module(Conv[dim](input_ch, out_channels, 3, padding=1)),
        )
        # self.use_fp16 = use_fp16

    def forward(self, x, timesteps, y=None):
        hs = []
        # print("device of timesteps, self.model_channels:", timesteps.device, self.model_channels)
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if y != None:
            text_outputs = self.token_embedding(y.float())
            emb = emb + text_outputs.to(emb)

        # print("forward, h = x.type(self.dtype), self.dtype =", self.dtype)
        h = x.type(self.dtype)
        # print("0,h.shape =", h.shape)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            # print("module encoder, h.shape =", h.shape)
        # print("2,h.shape =", h.shape)
        h = self.middle_block(h, emb)
        # print("middle block, h.shape =", h.shape)
        # print("2,h.shape =", h.shape)
        for module in self.output_blocks:
            # print("for module in self.output_blocks, h.shape =", h.shape)
            # print("len(hs) =", len(hs), ", hs[-1].shape =", hs[-1].shape)
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
            # print("module decoder, h.shape =", h.shape)

        # print("h = h.type(x.dtype), x.dtype =", x.dtype)
        h = h.type(x.dtype)
        h = self.out(h)
        # print("self.out(h)", "h.shape =", h.shape)

        return h 