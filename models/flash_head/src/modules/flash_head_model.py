import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
import torch.cuda.amp as amp
import torch.distributed as dist
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from xfuser.core.long_ctx_attention import xFuserLongContextAttention
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x

def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return torch.cat([f_freqs_cis, h_freqs_cis, w_freqs_cis], dim=1)


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor

def rope_apply(x, freqs, grid_sizes, use_usp=False, sp_size=1, sp_rank=0):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1) # [[N, head_dim/2], [N, head_dim/2], [N, head_dim/2]] # T H W 极坐标

    # loop over samples

    (f, h, w) = grid_sizes
    seq_len = f * h * w

    # precompute multipliers
    x_i = torch.view_as_complex(x[0, :s].to(torch.float64).reshape(
        s, n, -1, 2)) # [L, N, C/2] # 极坐标
    freqs_i = torch.cat([
        freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ],
                        dim=-1).reshape(seq_len, 1, -1) # seq_lens, 1,  3 * dim / 2 (T H W)

    if use_usp:
        # apply rotary embedding
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                        s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[0, s:]])
    else:
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[0, seq_len:]])
    return x_i.unsqueeze(0).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.use_usp = dist.is_initialized()
        self.sp_size = get_sequence_parallel_world_size() if self.use_usp else 1
        self.sp_rank = get_sequence_parallel_rank() if self.use_usp else 0

    def forward(self, x, freqs, grid_sizes):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x)

        if self.use_usp:
            from yunchang.kernels import AttnType
            if SAGE_ATTN_AVAILABLE:
                attn_type = AttnType.SAGE_AUTO
            else:
                attn_type = AttnType.FA

            x = xFuserLongContextAttention(attn_type=attn_type)(
                None,
                query=rope_apply(q, freqs, grid_sizes, self.use_usp, self.sp_size, self.sp_rank),
                key=rope_apply(k, freqs, grid_sizes, self.use_usp, self.sp_size, self.sp_rank),
                value=v.view(b, s, n, d),
            ).flatten(2)
        else:
            x = flash_attention(
                q=rope_apply(q, freqs, grid_sizes).flatten(2),
                k=rope_apply(k, freqs, grid_sizes).flatten(2),
                v=v,
                num_heads=self.num_heads
            )
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = flash_attention(q, k, v, num_heads=self.num_heads)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)

class DiTAudioBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6, i=0, num_layers=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.i = i
        self.num_layers = num_layers

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        self.use_usp = dist.is_initialized()
        self.sp_size = get_sequence_parallel_world_size() if self.use_usp else 1
        self.sp_rank = get_sequence_parallel_rank() if self.use_usp else 0

    def forward(self, x, context, t_mod, freqs, grid_sizes):
        e = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)

        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0], freqs, grid_sizes)

        x = x + y * e[2]

        x_1 = rearrange(self.norm3(x), 'b (f l) c -> (b f) l c', f=context.shape[1])
        context_1 = context.squeeze(0)

        if self.use_usp:
            context_1 = context_1.unsqueeze(1).repeat(1, self.sp_size, 1, 1).flatten(0,1)
            context_1 = torch.chunk(context_1, self.sp_size, dim=0)[self.sp_rank]

        x = x + self.cross_attn(x_1, context_1).flatten(0, 1).unsqueeze(0)

        y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
        x = x + y * e[5]

        return x

class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            t_mod(Tensor): Shape [B*21, C]
        """
        B, L, D = x.shape
        F = t_mod.shape[0] // B
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device).unsqueeze(1) + t_mod.unflatten(dim=0, sizes=(B, t_mod.shape[0]//B)).unsqueeze(2)).chunk(2, dim=2)

        x = rearrange(x, 'b (f l) d -> b f l d', f=F)
        x = (self.head(self.norm(x) * (1 + scale) + shift))
        x = rearrange(x, 'b f l d -> b (f l) d')
        return x

class WanModelAudioProject(ModelMixin, ConfigMixin):
    _no_split_modules = ['DiTAudioBlock']
    @register_to_config
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        vae_stride: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTAudioBlock(has_image_input, dim, num_heads, ffn_dim, eps, i, num_layers)
            for i in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        self.audio_emb = MLP(768, dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim)

        # init audio adapter
        audio_window = 5
        vae_scale = vae_stride[0]
        intermediate_dim = 512
        output_dim = 1536
        context_tokens = 32
        norm_output_audio = True
        self.audio_window = audio_window
        self.vae_scale = vae_scale
        self.audio_proj = AudioProjModel(
                    seq_len=audio_window,
                    seq_len_vf=audio_window+vae_scale-1,
                    intermediate_dim=intermediate_dim,
                    output_dim=output_dim,
                    context_tokens=context_tokens,
                    norm_output_audio=norm_output_audio,
                )

        self.use_usp = dist.is_initialized()
        self.sp_size = get_sequence_parallel_world_size() if self.use_usp else 1
        self.sp_rank = get_sequence_parallel_rank() if self.use_usp else 0
        
    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,  #(1, 16, 9, 64, 64))
                timestep: torch.Tensor, #(9,)
                context: torch.Tensor, #(5, 33, 12, 768)
                y: Optional[torch.Tensor] = None, #(1, 16, 9, 64, 64)
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):

        if self.freqs.device != x.device:
            self.freqs = self.freqs.to(x.device)

        x = torch.cat([x, y], dim=1) # (1, 32, 9, 64, 64)
        x, grid_sizes = self.patchify(x)
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.to(dtype=x.dtype)))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))   # (bsz, 6, 1536) 

        # ==================== 音频条件处理 ====================
        # 输入: context (bsz, 81, 5, 12, 768)
        # - 81 帧 = 1 (第一帧) + 80 (后续帧, 每4帧对应VAE压缩后的1帧)
        # - 5 是音频窗口大小 (audio_window)
        # - 12 是音频特征的 blocks
        # - 768 是音频特征维度
        
        audio_cond = context.to(device=x.device, dtype=x.dtype)
        
        # 1. 第一帧：直接使用完整的5帧音频窗口
        first_frame_audio = audio_cond[:, :1, ...]  # (bsz, 1, 5, 12, 768)
        
        # 2. 后续帧：需要根据帧位置选择不同的音频窗口
        # 将 32 帧重排为 (8 个 VAE latent, 每个4帧)
        latter_frames_audio = rearrange(
            audio_cond[:, 1:, ...], 
            "b (n_latent n_frame) w s c -> b n_latent n_frame w s c", 
            n_frame=self.vae_scale  # vae_scale=4
        )  # (bsz, 8, 4, 5, 12, 768)
        
        mid_idx = self.audio_window // 2  # 窗口中心索引: 5//2=2
        
        # 为每个 latent 的4帧选择合适的音频窗口:
        # - 第1帧 (帧索引0): 无过去，取前3帧窗口 [:mid_idx+1] = [:3]
        # - 中间帧 (帧索引1-2): 取中心1帧 [mid_idx:mid_idx+1] = [2:3]
        # - 第4帧 (帧索引3): 无未来，取后3帧窗口 [mid_idx:] = [2:]
        
        first_of_group = latter_frames_audio[:, :, :1, :mid_idx+1, ...]  # (bsz, 8, 1, 3, 12, 768)
        middle_of_group = latter_frames_audio[:, :, 1:-1, mid_idx:mid_idx+1, ...]  # (bsz, 8, 2, 1, 12, 768)
        last_of_group = latter_frames_audio[:, :, -1:, mid_idx:, ...]  # (bsz, 8, 1, 3, 12, 768)
        
        # 合并并展平窗口维度: (n_frame, window) -> (n_frame * window)
        latter_frames_audio_processed = torch.cat([
            rearrange(first_of_group, "b n_latent n_f w s c -> b n_latent (n_f w) s c"),
            rearrange(middle_of_group, "b n_latent n_f w s c -> b n_latent (n_f w) s c"),
            rearrange(last_of_group, "b n_latent n_f w s c -> b n_latent (n_f w) s c"),
        ], dim=2)  # (bsz, 8, 1*3 + 2*1 + 1*3, 12, 768) = (bsz, 8, 8, 12, 768)
        
        # 3. 通过 AudioProjModel 投影到 DiT 所需的特征空间
        context = self.audio_proj(
            first_frame_audio, 
            latter_frames_audio_processed
        ).to(x.dtype)  # (bsz, 9, 32, 1536)

        if self.use_usp:
            x = torch.chunk(x, self.sp_size, dim=1)[self.sp_rank]

        for block in self.blocks:
            x = block(x, context, t_mod, self.freqs, grid_sizes)
        x = self.head(x, t)   # (bsz, 9*32*32, 64)
        if self.use_usp:
            x = get_sp_group().all_gather(x, dim=1)
        x = self.unpatchify(x, grid_sizes)  # (bsz, 16, 21, 64, 64)
        return x


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
        audio_embeds_c = audio_embeds_c.view(batch_size_c*N_t, C_a)

        # second projection
        audio_embeds_c = torch.relu(self.proj2(audio_embeds_c))

        context_tokens = self.proj3(audio_embeds_c).reshape(batch_size_c*N_t, self.context_tokens, self.output_dim)

        # normalization and reshape
        with amp.autocast(dtype=torch.float32):
            context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(context_tokens, "(bz f) m c -> bz f m c", f=video_length)

        return context_tokens