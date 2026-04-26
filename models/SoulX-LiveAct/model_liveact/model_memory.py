# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import copy
import math
import numpy as np
import os
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin

from .attention import flash_attention, SingleStreamAttention, sdpa_attention, flex_attention
from fp8_gemm import FP8Linear
import logging

try:
    from sageattention import sageattn

    USE_SAGEATTN = True
    logging.info("Using sageattn")
except:
    USE_SAGEATTN = False

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


# @amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    s, n, c = x.size(1), x.size(2), x.size(3) // 2

    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = s
        f = int(seq_len // (h * w))
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)
        freqs_i = freqs_i.to(device=x_i.device)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        output.append(x_i)
    return torch.stack(output)  # .float()


def rope_apply(x, grid_sizes, freqs, f_list=[], rope_list=[]):
    s, n, c = x.size(1), x.size(2), x.size(3) // 2

    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for f_l, r_l in zip(f_list, rope_list):
        start_f, end_f = f_l
        start_r, end_r = r_l
        f = end_f - start_f
        _, h, w = grid_sizes.tolist()[0]
        seq_len = (end_f - start_f) * h * w
        x_i = torch.view_as_complex(
            x[0, start_f * h * w:end_f * h * w].to(torch.float64) \
                .reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat([
            freqs[0][start_r:end_r].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)
        freqs_i = freqs_i.to(device=x_i.device)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        output.append(x_i)
    return torch.concat(output, dim=0).unsqueeze(0)


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).to(dtype=x.dtype) * self.weight.to(dtype=x.dtype)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        out = F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            None if self.weight is None else self.weight.float(),
            None if self.bias is None else self.bias.float(),
            self.eps
        ).to(origin_dtype)
        return out


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attn_mask = None
        self.frame_seqlen = None
        self.memory_proj_k = nn.Conv1d(self.dim, self.dim, kernel_size=5, stride=5, groups=self.dim, bias=False)
        self.memory_proj_v = nn.Conv1d(self.dim, self.dim, kernel_size=5, stride=5, groups=self.dim, bias=False)

    def post_init(self, device):
        self.memory_proj_k = nn.Conv1d(self.dim, self.dim, kernel_size=5, stride=5, groups=self.dim, bias=False).to(
            device, dtype=torch.bfloat16)
        self.memory_proj_v = nn.Conv1d(self.dim, self.dim, kernel_size=5, stride=5, groups=self.dim, bias=False).to(
            device, dtype=torch.bfloat16)
        nn.init.constant_(self.memory_proj_k.weight, 1.0 / 5.0)
        nn.init.constant_(self.memory_proj_v.weight, 1.0 / 5.0)

    def k_compress(self, k, n_frame=5):
        B, N, H, C = k.shape
        assert N % n_frame == 0
        T = N // n_frame
        k = k.view(B, N, H * C).transpose(1, 2)
        k = self.memory_proj_k(k)
        k = k.view(B, H, C, T).permute(0, 3, 1, 2)
        return k

    def v_compress(self, v, n_frame=5):
        B, N, H, C = v.shape
        assert N % n_frame == 0
        T = N // n_frame
        v = v.view(B, N, H * C).transpose(1, 2)
        v = self.memory_proj_k(v)
        v = v.view(B, H, C, T).permute(0, 3, 1, 2)
        return v

    def kv_mean(self, kv, n_frame=5):
        B, N, H, C = kv.shape
        assert N % n_frame == 0
        T = N // n_frame
        kv = kv.view(B, T, n_frame, H, C).mean(dim=2)
        return kv

    def init_kvidx(self, frame_len, world_size):
        self.frame_seqlen = frame_len
        self.kv_idx0 = torch.tensor(list(range(6 * frame_len // world_size)),
                                    device=f'cuda:{int(os.getenv("RANK", 0))}')
        self.kv_idx2 = torch.tensor(list(range(14 * frame_len // world_size)),
                                    device=f'cuda:{int(os.getenv("RANK", 0))}')

    def _move_kv_cache_to_device(self, kv_cache, device):
        kv_cache["k"] = kv_cache["k"].to(device=device, non_blocking=True)
        kv_cache["v"] = kv_cache["v"].to(device=device, non_blocking=True)
        if kv_cache.get("k_scale") is not None:
            kv_cache["k_scale"] = kv_cache["k_scale"].to(device=device, non_blocking=True)
        if kv_cache.get("v_scale") is not None:
            kv_cache["v_scale"] = kv_cache["v_scale"].to(device=device, non_blocking=True)

    def _quantize_kv_tensor(self, kv):
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale = kv.detach().abs().amax(dim=-1, keepdim=True).to(torch.float32)
        scale = torch.clamp(scale / fp8_max, min=1e-12)
        q_kv = (kv / scale.to(dtype=kv.dtype)).to(torch.float8_e4m3fn)
        return q_kv.contiguous(), scale.contiguous()

    def _dequantize_kv_tensor(self, q_kv, scale, dtype):
        return q_kv.to(dtype=dtype) * scale.to(device=q_kv.device, dtype=dtype)

    def _load_kv_cache(self, kv_cache, device, dtype):
        if kv_cache["offload_cache"]:
            self._move_kv_cache_to_device(kv_cache, device)

        if kv_cache.get("fp8_kv_cache", False):
            k_cache = self._dequantize_kv_tensor(kv_cache["k"], kv_cache["k_scale"], dtype)
            v_cache = self._dequantize_kv_tensor(kv_cache["v"], kv_cache["v_scale"], dtype)
        else:
            if kv_cache["k"].dtype != dtype:
                kv_cache["k"] = kv_cache["k"].to(dtype=dtype)
            if kv_cache["v"].dtype != dtype:
                kv_cache["v"] = kv_cache["v"].to(dtype=dtype)
            k_cache = kv_cache["k"]
            v_cache = kv_cache["v"]
        return k_cache, v_cache

    def _store_kv_cache(self, kv_cache, k_cache, v_cache):
        if kv_cache.get("fp8_kv_cache", False):
            kv_cache["k"], kv_cache["k_scale"] = self._quantize_kv_tensor(k_cache)
            kv_cache["v"], kv_cache["v_scale"] = self._quantize_kv_tensor(v_cache)
        else:
            kv_cache["k"] = k_cache
            kv_cache["v"] = v_cache

        if kv_cache["offload_cache"]:
            self._move_kv_cache_to_device(kv_cache, 'cpu')

    def forward(self, x, seq_lens, grid_sizes, freqs, kv_cache={}, start_idx=None, end_idx=None, update_cache=False):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        k_cache, v_cache = self._load_kv_cache(kv_cache, f'cuda:{int(os.getenv("RANK", 0))}', torch.bfloat16)

        frame_seqlen = self.frame_seqlen
        if frame_seqlen is None:
            raise RuntimeError("WanSelfAttention.init_kvidx() must be called before forward().")
        current_start_frame = start_idx // frame_seqlen

        if update_cache:
            if kv_cache["mean_memory"]:
                k_compress, v_compress = self.kv_mean, self.kv_mean
            else:
                k_compress, v_compress = self.k_compress, self.v_compress
            k_cache[:, 2 * frame_seqlen: 3 * frame_seqlen].copy_(
                k_compress(k_cache[:, 2 * frame_seqlen: 7 * frame_seqlen]))
            v_cache[:, 2 * frame_seqlen: 3 * frame_seqlen].copy_(
                v_compress(v_cache[:, 2 * frame_seqlen: 7 * frame_seqlen]))
            k_cache[:, 3 * frame_seqlen: 4 * frame_seqlen].copy_(
                k_compress(k_cache[:, 7 * frame_seqlen: 12 * frame_seqlen]))
            v_cache[:, 3 * frame_seqlen: 4 * frame_seqlen].copy_(
                v_compress(v_cache[:, 7 * frame_seqlen: 12 * frame_seqlen]))

            k_cache[:, 4 * frame_seqlen: 6 * frame_seqlen].copy_(k_cache[:, 12 * frame_seqlen: 14 * frame_seqlen])
            v_cache[:, 4 * frame_seqlen: 6 * frame_seqlen].copy_(v_cache[:, 12 * frame_seqlen: 14 * frame_seqlen])

        if start_idx != 0:
            k_cache[:, 6 * frame_seqlen:] = k
            v_cache[:, 6 * frame_seqlen:] = v
        else:
            k_cache[:, : 6 * frame_seqlen] = k
            v_cache[:, : 6 * frame_seqlen] = v

        roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
        roped_key = causal_rope_apply(k_cache, grid_sizes, freqs, start_frame=0).type_as(v)

        if USE_SAGEATTN:
            x = sageattn(
                roped_query,
                roped_key[:, :end_idx, ...],
                v_cache[:, :end_idx, ...],
                tensor_layout="NHD",
                is_causal=False,
            ).type_as(x)
        else:
            x = sdpa_attention(
                q=roped_query,
                k=roped_key[:, :end_idx, ...],
                v=v_cache[:, :end_idx, ...],
                k_lens=seq_lens,
                window_size=self.window_size,
                attn_mask=self.attn_mask,
            ).type_as(x)

        self._store_kv_cache(kv_cache, k_cache, v_cache)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x, None


class WanI2VCrossAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens, cross_kv_cache={}):
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        # if not cross_kv_cache:
        #     # print('----init cross_kv_cache!!!')
        #     k = self.norm_k(self.k(context)).view(b, -1, n, d)
        #     v = self.v(context).view(b, -1, n, d)
        #     k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        #     v_img = self.v_img(context_img).view(b, -1, n, d)
        #     cross_kv_cache['k'], cross_kv_cache['v'], cross_kv_cache['k_img'], cross_kv_cache['v_img'] = \
        #         k, v, k_img, v_img
        # else:
        #     # print('----use cross_kv_cache!!!')
        #     k, v, k_img, v_img = \
        #         cross_kv_cache['k'], cross_kv_cache['v'], cross_kv_cache['k_img'], cross_kv_cache['v_img']
        if USE_SAGEATTN:
            img_x = sageattn(q, k_img, v_img, tensor_layout='NHD')
            x = sageattn(q, k, v, tensor_layout='NHD')
        else:
            # img_x = flash_attention(q, k_img, v_img, k_lens=None)
            img_x = sdpa_attention(q, k_img, v_img, k_lens=None)
            # compute attention
            # x = flash_attention(q, k, v, k_lens=context_lens)
            x = sdpa_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 output_dim=768,
                 norm_input_visual=True,
                 class_range=24,
                 class_interval=4):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanI2VCrossAttention(dim,
                                               num_heads,
                                               (-1, -1),
                                               qk_norm,
                                               eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

        # init audio module
        self.audio_cross_attn = SingleStreamAttention(
            dim=dim,
            encoder_hidden_states_dim=output_dim,
            num_heads=num_heads,
            qk_norm=False,
            qkv_bias=True,
            eps=eps,
            norm_layer=WanRMSNorm,
            # class_range=class_range,
            # class_interval=class_interval
        )
        self.norm_x = WanLayerNorm(dim, eps, elementwise_affine=True) if norm_input_visual else nn.Identity()

    def forward(
            self,
            x,
            e,
            seq_lens,
            grid_sizes,
            freqs,
            context,
            context_lens,
            kv_cache={},
            start_idx=None,
            end_idx=None,
            update_cache=False,
            cross_kv_cache={},
            audio_embedding=None,
            ref_target_masks=None,
            human_num=None,
            skip_audio=False,
    ):

        dtype = x.dtype
        # assert e.dtype == torch.float32
        if len(e.shape) == 3:
            # with amp.autocast(dtype=torch.float32):
            e = (self.modulation.to(e.device) + e).chunk(6, dim=1)
        else:
            # with amp.autocast(dtype=torch.float32):
            e = (self.modulation.unsqueeze(-2).to(e.device) + e)[0].chunk(6, dim=0)
        # assert e[0].dtype == torch.float32

        # self-attention
        y, x_ref_attn_map = self.self_attn(
            (self.norm1(x).float() * (1 + e[1]) + e[0]).type_as(x), seq_lens, grid_sizes,
            freqs, kv_cache=kv_cache, start_idx=start_idx, end_idx=end_idx, update_cache=update_cache)
        # with amp.autocast(dtype=torch.float32):
        x = x + y * e[2]

        x = x.to(dtype)

        # cross-attention of text
        x = x + self.cross_attn(self.norm3(x), context, context_lens, cross_kv_cache=cross_kv_cache)

        # cross attn of audio
        if not skip_audio:
            frame_seqlen = self.self_attn.frame_seqlen
            start_f = start_idx // frame_seqlen
            x_a = self.audio_cross_attn(self.norm_x(x), encoder_hidden_states=audio_embedding,
                                        frame_seqlen=frame_seqlen, start_f=start_f, USE_SAGEATTN=USE_SAGEATTN)
            if start_f == 0:
                x_a[:, :frame_seqlen] = 0
            x = x + x_a

        y = self.ffn((self.norm2(x).float() * (1 + e[4]) + e[3]).to(dtype))
        # with amp.autocast(dtype=torch.float32):
        x = x + y * e[5]

        x = x.to(dtype)

        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.to(e.device) + e.unsqueeze(1)).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class AudioProjModel(ModelMixin, ConfigMixin):
    def __init__(
            self,
            seq_len=5,
            seq_len_vf=12,
            blocks=12,
            channels=768,
            intermediate_dim=512,
            output_dim=768,
            context_tokens=32,
            norm_output_audio=False,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels
        self.input_dim_vf = seq_len_vf * blocks * channels
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj1_vf = nn.Linear(self.input_dim_vf, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.norm = nn.LayerNorm(output_dim) if norm_output_audio else nn.Identity()

    def forward(self, audio_embeds, audio_embeds_vf):
        video_length = audio_embeds.shape[1] + audio_embeds_vf.shape[1]
        B, _, _, S, C = audio_embeds.shape

        # process audio of first frame
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        # process audio of latter frame
        audio_embeds_vf = rearrange(audio_embeds_vf, "bz f w b c -> (bz f) w b c")
        batch_size_vf, window_size_vf, blocks_vf, channels_vf = audio_embeds_vf.shape
        audio_embeds_vf = audio_embeds_vf.view(batch_size_vf, window_size_vf * blocks_vf * channels_vf)

        # first projection
        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds_vf = torch.relu(self.proj1_vf(audio_embeds_vf))
        audio_embeds = rearrange(audio_embeds, "(bz f) c -> bz f c", bz=B)
        audio_embeds_vf = rearrange(audio_embeds_vf, "(bz f) c -> bz f c", bz=B)
        audio_embeds_c = torch.concat([audio_embeds, audio_embeds_vf], dim=1)
        batch_size_c, N_t, C_a = audio_embeds_c.shape
        audio_embeds_c = audio_embeds_c.view(batch_size_c * N_t, C_a)

        # second projection
        audio_embeds_c = torch.relu(self.proj2(audio_embeds_c))

        context_tokens = self.proj3(audio_embeds_c).reshape(batch_size_c * N_t, self.context_tokens, self.output_dim)

        # normalization and reshape
        # with amp.autocast(dtype=torch.float32):
        context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(context_tokens, "(bz f) m c -> bz f m c", f=video_length)

        return context_tokens


from torch.utils.checkpoint import checkpoint


class WanBlockOffloadManager:
    def __init__(self, blocks, onload_device, offload_device='cpu'):
        self.blocks = blocks
        self.onload_device = torch.device(onload_device)
        self.offload_device = torch.device(offload_device)
        self.prefetch_stream = torch.cuda.Stream(device=self.onload_device)
        self.compute_slot = 0
        self.prefetch_slot = 1
        self.pending_slots = set()
        self.slot_block_indices = [None, None]
        self.cuda_blocks = nn.ModuleList([
            copy.deepcopy(self.blocks[0]).to(self.onload_device),
            copy.deepcopy(self.blocks[0]).to(self.onload_device),
        ])

        for block in self.blocks:
            block.to(self.offload_device)
            self._pin_module_memory(block)

    def _copy_tensor(self, dst, src):
        dst.copy_(src, non_blocking=True)

    def _pin_tensor(self, tensor):
        if tensor is None or tensor.device.type != 'cpu' or tensor.is_pinned():
            return tensor
        return tensor.pin_memory()

    def _pin_module_memory(self, module):
        for name, param in module.named_parameters(recurse=False):
            if param is not None:
                param.data = self._pin_tensor(param.data)

        for name, buffer in module.named_buffers(recurse=False):
            if buffer is not None:
                module._buffers[name] = self._pin_tensor(buffer)

        if isinstance(module, FP8Linear):
            module._fp16_weight_cpu = self._pin_tensor(module._fp16_weight_cpu)
            module._fp16_bias_cpu = self._pin_tensor(module._fp16_bias_cpu)

        for child in module.children():
            self._pin_module_memory(child)

    def _copy_fp8_linear(self, dst_module, src_module):
        if dst_module.linear is not None and src_module.linear is not None:
            self._copy_module_state(dst_module.linear, src_module.linear)

        if dst_module.bias is not None and src_module.bias is not None:
            self._copy_tensor(dst_module.bias.data, src_module.bias.data)

        dst_module._fp16_weight_cpu = src_module._fp16_weight_cpu
        dst_module._fp16_bias_cpu = src_module._fp16_bias_cpu

        if src_module._fp8_weight is None or src_module._fp8_weight_scale is None:
            dst_module._fp8_weight = None
            dst_module._fp8_weight_scale = None
            dst_module._weight_cache_device = None
            if dst_module._fp16_weight_cpu is not None:
                dst_module.materialize_fp8_weight(self.onload_device)
        else:
            if dst_module._fp8_weight is None or dst_module._fp8_weight.shape != src_module._fp8_weight.shape:
                dst_module._fp8_weight = src_module._fp8_weight.to(device=self.onload_device, non_blocking=True)
            else:
                self._copy_tensor(dst_module._fp8_weight, src_module._fp8_weight)

            if dst_module._fp8_weight_scale is None or dst_module._fp8_weight_scale.shape != src_module._fp8_weight_scale.shape:
                dst_module._fp8_weight_scale = src_module._fp8_weight_scale.to(device=self.onload_device,
                                                                               non_blocking=True)
            else:
                self._copy_tensor(dst_module._fp8_weight_scale, src_module._fp8_weight_scale)
            dst_module._weight_cache_device = dst_module._cached_fp8_device()

        dst_module._last_weight_version = src_module._last_weight_version

    def _copy_module_state(self, dst_module, src_module):
        if isinstance(dst_module, FP8Linear) and isinstance(src_module, FP8Linear):
            self._copy_fp8_linear(dst_module, src_module)
            return

        dst_params = dict(dst_module.named_parameters(recurse=False))
        src_params = dict(src_module.named_parameters(recurse=False))
        for name, dst_param in dst_params.items():
            src_param = src_params.get(name)
            if src_param is not None:
                self._copy_tensor(dst_param.data, src_param.data)

        dst_buffers = dict(dst_module.named_buffers(recurse=False))
        src_buffers = dict(src_module.named_buffers(recurse=False))
        for name, dst_buffer in dst_buffers.items():
            src_buffer = src_buffers.get(name)
            if src_buffer is not None:
                self._copy_tensor(dst_buffer, src_buffer)

        dst_children = dict(dst_module.named_children())
        src_children = dict(src_module.named_children())
        for name, dst_child in dst_children.items():
            src_child = src_children.get(name)
            if src_child is not None:
                self._copy_module_state(dst_child, src_child)

        if hasattr(src_module, "frame_seqlen"):
            dst_module.frame_seqlen = src_module.frame_seqlen
        if hasattr(src_module, "kv_idx0"):
            dst_module.kv_idx0 = src_module.kv_idx0
        if hasattr(src_module, "kv_idx2"):
            dst_module.kv_idx2 = src_module.kv_idx2

    def _load_slot(self, slot_idx, block_idx, async_transfer=False):
        def copy_block():
            self._copy_module_state(self.cuda_blocks[slot_idx], self.blocks[block_idx])
            self.slot_block_indices[slot_idx] = block_idx

        if async_transfer:
            with torch.cuda.stream(self.prefetch_stream):
                copy_block()
            self.pending_slots.add(slot_idx)
        else:
            copy_block()
            self.pending_slots.discard(slot_idx)

    def _wait_slot(self, slot_idx):
        if slot_idx in self.pending_slots:
            torch.cuda.current_stream(device=self.onload_device).wait_stream(self.prefetch_stream)
            self.pending_slots.discard(slot_idx)

    def get_block(self, block_idx):
        if self.slot_block_indices[self.compute_slot] == block_idx:
            self._wait_slot(self.compute_slot)
        elif self.slot_block_indices[self.prefetch_slot] == block_idx:
            self._wait_slot(self.prefetch_slot)
            self.compute_slot, self.prefetch_slot = self.prefetch_slot, self.compute_slot
        else:
            self._load_slot(self.compute_slot, block_idx, async_transfer=False)

        next_idx = block_idx + 1
        if next_idx < len(self.blocks) and self.slot_block_indices[self.prefetch_slot] != next_idx:
            # We are about to overwrite self.prefetch_slot on the prefetch stream.
            # Must ensure the compute stream has finished using it from previous steps.
            self.prefetch_stream.wait_stream(torch.cuda.current_stream(device=self.onload_device))
            self._load_slot(self.prefetch_slot, next_idx, async_transfer=True)

        return self.cuda_blocks[self.compute_slot]

    def unload_all(self):
        torch.cuda.current_stream(device=self.onload_device).wait_stream(self.prefetch_stream)
        self.pending_slots.clear()
        self.slot_block_indices = [None, None]


class WanModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='i2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 # audio params
                 audio_window=5,
                 intermediate_dim=512,
                 output_dim=768,
                 context_tokens=32,
                 vae_scale=4,  # vae timedownsample scale

                 norm_input_visual=True,
                 norm_output_audio=True,
                 weight_init=True):
        super().__init__()

        assert model_type == 'i2v', 'MultiTalk model requires your model_type is i2v.'
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.gradient_checkpointing = False

        self.norm_output_audio = norm_output_audio
        self.audio_window = audio_window
        self.intermediate_dim = intermediate_dim
        self.vae_scale = vae_scale

        self.return_layers_cosine = False
        self.cos_sims = []
        self.skip_layer = []
        self.block_offload_manager = None
        self.block_offload_enabled = False

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps,
                              output_dim=output_dim, norm_input_visual=norm_input_visual)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)
        else:
            raise NotImplementedError('Not supported model type.')

        # init audio adapter
        self.audio_proj = AudioProjModel(
            seq_len=audio_window,
            seq_len_vf=audio_window + vae_scale - 1,
            intermediate_dim=intermediate_dim,
            output_dim=output_dim,
            context_tokens=context_tokens,
            norm_output_audio=norm_output_audio,
        )

        # initialize weights
        if weight_init:
            self.init_weights()

    def init_freqs(self):
        d = self.dim // self.num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

    def enable_block_offload(self, onload_device=None, offload_device='cpu'):
        if onload_device is None:
            onload_device = self.patch_embedding.weight.device
        onload_device = torch.device(onload_device)
        if onload_device.type != 'cuda':
            raise ValueError("WanModel block offload requires a CUDA onload device.")

        self.block_offload_manager = WanBlockOffloadManager(
            self.blocks,
            onload_device=onload_device,
            offload_device=offload_device,
        )
        self.block_offload_enabled = True
        torch.cuda.empty_cache()
        return self

    def forward(
            self,
            x,
            t,
            context,
            seq_len=None,
            clip_fea=None,
            y=None,
            audio=None,
            ref_target_masks=None,
            e0=None,
            kv_cache={},
            start_idx=None,
            end_idx=None,
            cross_kv_cache={},
            update_cache=True,
            skip_audio=False,
    ):
        assert clip_fea is not None and y is not None

        _, T, H, W = x[0].shape
        N_t = T // self.patch_size[0]
        N_h = H // self.patch_size[1]
        N_w = W // self.patch_size[2]

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        x[0] = x[0].to(context[0].dtype)

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        x = torch.cat(x)

        # time embeddings
        if e0 is None:
            # with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            # assert e.dtype == torch.float32 and e0.dtype == torch.float32
        else:
            # with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())

        # text embedding
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # clip embedding
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1).to(x.dtype)

        audio_cond = audio.to(device=x.device, dtype=x.dtype)
        first_frame_audio_emb_s = audio_cond[:, :1, ...]
        latter_frame_audio_emb = audio_cond[:, 1:, ...]
        latter_frame_audio_emb = rearrange(latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=self.vae_scale)
        middle_index = self.audio_window // 2
        latter_first_frame_audio_emb = latter_frame_audio_emb[:, :, :1, :middle_index + 1, ...]
        latter_first_frame_audio_emb = rearrange(latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
        latter_last_frame_audio_emb = latter_frame_audio_emb[:, :, -1:, middle_index:, ...]
        latter_last_frame_audio_emb = rearrange(latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
        latter_middle_frame_audio_emb = latter_frame_audio_emb[:, :, 1:-1, middle_index:middle_index + 1, ...]
        latter_middle_frame_audio_emb = rearrange(latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
        latter_frame_audio_emb_s = torch.concat(
            [latter_first_frame_audio_emb, latter_middle_frame_audio_emb, latter_last_frame_audio_emb], dim=2)
        audio_embedding = self.audio_proj(first_frame_audio_emb_s, latter_frame_audio_emb_s)
        human_num = len(audio_embedding)
        audio_embedding = torch.concat(audio_embedding.split(1), dim=2).to(x.dtype)

        # convert ref_target_masks to token_ref_target_masks
        if ref_target_masks is not None:
            ref_target_masks = ref_target_masks.unsqueeze(0)  # .to(torch.float32)
            token_ref_target_masks = nn.functional.interpolate(ref_target_masks, size=(N_h, N_w), mode='nearest')
            token_ref_target_masks = token_ref_target_masks.squeeze(0)
            token_ref_target_masks = (token_ref_target_masks > 0)
            token_ref_target_masks = token_ref_target_masks.view(token_ref_target_masks.shape[0], -1)
            token_ref_target_masks = token_ref_target_masks.to(x.dtype)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            audio_embedding=audio_embedding,
            ref_target_masks=token_ref_target_masks,
            human_num=human_num,
            start_idx=start_idx,
            end_idx=end_idx,
            update_cache=update_cache,
            skip_audio=skip_audio,
        )

        block_offload_manager = self.block_offload_manager if self.block_offload_enabled else None
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block_index, block in enumerate(self.blocks):
                if block_offload_manager is not None:
                    block = block_offload_manager.get_block(block_index)
                if kv_cache.get(block_index) is None: kv_cache[block_index] = {}
                if cross_kv_cache.get(block_index) is None: cross_kv_cache[block_index] = {}
                x = checkpoint(
                    block, x, kv_cache=kv_cache[block_index], cross_kv_cache=cross_kv_cache[block_index],
                    use_reentrant=False, **kwargs
                )
        else:
            for block_index, block in enumerate(self.blocks):
                if block_offload_manager is not None:
                    block = block_offload_manager.get_block(block_index)
                if kv_cache.get(block_index) is None: kv_cache[block_index] = {}
                if cross_kv_cache.get(block_index) is None: cross_kv_cache[block_index] = {}
                x = block(x, kv_cache=kv_cache[block_index], cross_kv_cache=cross_kv_cache[block_index], **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)

        return torch.stack(x)  # .float()

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
