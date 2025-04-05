# CViT (Continuous Vision Transformer) - PyTorch Version (1:1 with Flax)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# === Positional Embeddings ===
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)

    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb

def get_1d_sincos_pos_embed(embed_dim, length):
    pos = torch.arange(length, dtype=torch.float32)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos).unsqueeze(0)

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        return torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)

    grid_h = torch.arange(grid_size[0], dtype=torch.float32)
    grid_w = torch.arange(grid_size[1], dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")  # Flax: W then H
    grid = torch.stack([grid_h.reshape(-1), grid_w.reshape(-1)], dim=0)  # (2, H*W)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed.unsqueeze(0)  # (1, H*W, D)

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=(1, 16, 16), emb_dim=768, use_norm=False, layer_norm_eps=1e-5):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.use_norm = use_norm

        self.proj = nn.Conv3d(
            in_channels=4,               # assuming 4-channel input
            out_channels=emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        if use_norm:
            self.norm = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        # x: (B, T, H, W, C)
        b, t, h, w, c = x.shape
        x = x.permute(0, 4, 1, 2, 3)  # → (B, C, T, H, W)
        x = self.proj(x)             # → (B, D, T', H', W')
        _, d, t_p, h_p, w_p = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(b, t_p, h_p * w_p, d)  # (B, T', S, D)
        x = self.norm(x)
        return x

class MlpBlock(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(out_dim, hidden_dim)  # Matches self.dim in Flax
        self.fc2 = nn.Linear(hidden_dim, out_dim)     # Matches self.out_dim in Flax

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

class Mlp(nn.Module):
    def __init__(self, num_layers, hidden_dim, out_dim, eps=1e-5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU()
            ) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim, eps) for _ in range(num_layers)
        ])
        self.final = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            y = layer(x)
            x = norm(x + y)
        return self.final(x)

class SelfAttnBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_ratio, layer_norm_eps=1e-5):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.mlp = MlpBlock(emb_dim * mlp_ratio, emb_dim)

    def forward(self, x):
        # x: (B, N, D)
        x1 = self.norm1(x)
        attn_output, _ = self.attn(x1, x1, x1)
        print(f'x shape {x.shape}')
        x = x + attn_output  # residual

        x2 = self.norm2(x)
        x = x + self.mlp(x2)  # second residual

        return x

class CrossAttnBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_ratio, layer_norm_eps=1e-5):
        super().__init__()
        self.norm_q = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.norm_kv = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.mlp = MlpBlock(emb_dim * mlp_ratio, emb_dim)

    def forward(self, q_input, kv_input):
        # Normalize
        q = self.norm_q(q_input)
        kv = self.norm_kv(kv_input)

        # Flatten B, S for TimeAggregation shape (B, S, T, D)
        if q.dim() == 4:
            bq, sq, tq, d = q.shape
            bk, sk, tk, _ = kv.shape
            q = q.reshape(bq * sq, tq, d)
            kv = kv.reshape(bk * sk, tk, d)
            x, _ = self.attn(q, kv, kv)
            x = x + q_input.reshape(bq * sq, tq, d)
            x = x.reshape(bq, sq, tq, d)
        else:
            # Normal (B, N, D) inputs for decoder cross-attn
            x, _ = self.attn(q, kv, kv)
            x = x + q_input

        # FFN
        x2 = self.norm2(x)
        print(f'x shape {x.shape}')
        x2 = self.mlp(x2)
        return x + x2


class TimeAggregation(nn.Module):
    def __init__(self, emb_dim, depth, num_latents=1, num_heads=8, mlp_ratio=1, eps=1e-5):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, emb_dim))
        self.blocks = nn.ModuleList([
            CrossAttnBlock(emb_dim, num_heads, mlp_ratio, eps) for _ in range(depth)
        ])

    def forward(self, x):  # (B, T, S, D)
        b, t, s, d = x.shape
        latents = repeat(self.latents, 't d -> b s t d', b=b, s=s).to(x.device)
        x = rearrange(x, 'b t s d -> b s t d')
        for block in self.blocks:
            latents = block(latents, x)
        return rearrange(latents, 'b s t d -> b t s d')

class Encoder(nn.Module):
    def __init__(self, patch_size, emb_dim, depth, num_heads, mlp_ratio, eps=1e-5,
                 input_time=6, input_hw=(128, 384)):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size, emb_dim)
        self.time_agg = TimeAggregation(emb_dim, depth=2, num_latents=1,
                                        num_heads=num_heads, mlp_ratio=mlp_ratio, eps=eps)
        self.norm = nn.LayerNorm(emb_dim, eps)
        self.blocks = nn.ModuleList([
            SelfAttnBlock(emb_dim, num_heads, mlp_ratio, eps) for _ in range(depth)
        ])
        self.emb_dim = emb_dim
        self.patch_size = patch_size

        # === Match Flax-style positional embeddings: register them once and reuse ===
        pt, ph, pw = patch_size
        h_p, w_p = input_hw[0] // ph, input_hw[1] // pw
        t_p = input_time // pt

        t_emb = get_1d_sincos_pos_embed(emb_dim, t_p)         # (1, T', D)
        s_emb = get_2d_sincos_pos_embed(emb_dim, (h_p, w_p))  # (1, S, D)

        self.register_buffer("t_emb", t_emb, persistent=True)
        self.register_buffer("s_emb", s_emb, persistent=True)

    def forward(self, x):
        # x: (B, T, H, W, C)
        x = self.patch_embed(x)  # (B, T', S, D)
        x = x + self.t_emb[:, :, None, :] + self.s_emb[:, None, :, :]  # (B, T', S, D)

        x = self.time_agg(x)
        x = self.norm(x)
        x = rearrange(x, 'b t s d -> b (t s) d')
        for block in self.blocks:
            x = block(x)
        return x


class CViT(nn.Module):
    def __init__(self, patch_size=(1, 16, 16), grid_size=(128, 384), latent_dim=256,
                 emb_dim=256, depth=3, num_heads=8, dec_emb_dim=256, dec_num_heads=8,
                 dec_depth=1, num_mlp_layers=1, mlp_ratio=1, out_dim=4, eps=1e5,
                 embedding_type='grid', layer_norm_eps=1e-5):
        super().__init__()
        self.encoder = Encoder(patch_size, emb_dim, depth, num_heads, mlp_ratio, eps=layer_norm_eps)
        self.embedding_type = embedding_type
        self.dec_emb_dim = dec_emb_dim
        self.eps = eps
        self.patch_size = patch_size
        self.grid_size = grid_size

        if embedding_type == "grid":
            x = torch.linspace(0, 1, grid_size[0])
            y = torch.linspace(0, 1, grid_size[1])
            xx, yy = torch.meshgrid(x, y, indexing="ij")
            self.register_buffer("grid", torch.stack([xx.flatten(), yy.flatten()], dim=1))
            self.latents = nn.Parameter(torch.randn(grid_size[0] * grid_size[1], latent_dim))

        self.coords_proj = nn.Linear(dec_emb_dim, dec_emb_dim)
        self.cross_attn = nn.ModuleList([
            CrossAttnBlock(dec_emb_dim, dec_num_heads, mlp_ratio, layer_norm_eps)
            for _ in range(dec_depth)
        ])
        self.norm = nn.LayerNorm(dec_emb_dim, eps=layer_norm_eps)
        self.mlp = Mlp(num_layers=num_mlp_layers,
                       hidden_dim=dec_emb_dim,
                       out_dim=out_dim,
                       eps=layer_norm_eps)
        self.enc_to_dec = nn.Linear(emb_dim, dec_emb_dim)

    def forward(self, x, coords=None, train=False):
        b, t, h, w, c = x.shape
        print(f'input shape, {x.shape}')
        print(f'grid shape, {self.grid_size}')
        if coords is not None:
            print(f'coords shape {coords.shape}')
        if coords is None:
            x_lin = torch.linspace(0, 1, h, device=x.device)
            y_lin = torch.linspace(0, 1, w, device=x.device)
            xx, yy = torch.meshgrid(x_lin, y_lin, indexing='ij')
            coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # (H*W, 2)

        x = self.encoder(x)                # → (B, N, D)
        x = F.layer_norm(x, (x.shape[-1],))  # Matches Flax LN after encoder
        print(f'shape before dec: {x.shape}')
        x = self.enc_to_dec(x)             # → (B, N, D)
        print(f'shape after dec: {x.shape}')

        if self.embedding_type == "grid":
            d2 = ((coords[:, None, :] - self.grid[None, :, :]) ** 2).sum(dim=-1)
            w = torch.exp(-self.eps * d2)
            w = w / w.sum(dim=1, keepdim=True)
            coords = torch.einsum("ic,pi->pc", self.latents, w)
            coords = self.coords_proj(coords)
            coords = F.layer_norm(coords, (self.dec_emb_dim,))
            coords = repeat(coords, 'n d -> b n d', b=b)

        for block in self.cross_attn:
            coords = block(coords, x)

        x = self.norm(coords)
        x = self.mlp(x)  # Final prediction (B, N, out_dim)
        #print(h, w)
        if not train:
            x = rearrange(x, "b (h w) c -> b h w c", h=self.grid_size[0], w=self.grid_size[1])
        return x
