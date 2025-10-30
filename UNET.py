import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import math


# ==============================
#  CONFIG
# ==============================
class Config:
    vocab_size = 100
    emb_size = 512
    max_len = 512
    num_heads = 16
    num_layers = 12
    dropout = 0.1
    in_channels = 4
    out_channels = 4
    img_size = 32
    time_emb_dim = 512
    features = [64, 128, 256]
    cross_attention_levels = [1, 2]


config = Config()


# ==============================
#  TEXT ENCODER
# ==============================
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=10000):
        super().__init__()
        self.dim = dim
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, length):
        # length: int
        return self.pe[:length]  # returns [length, dim]

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(config.vocab_size, config.emb_size)
        self.pos_emb = SinusoidalPositionalEmbedding(config.emb_size, max_len=config.max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=config.emb_size,
            nhead=config.num_heads,
            dim_feedforward=config.emb_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)

    def forward(self, text_ids):
        # text_ids: [B, L]
        B, L = text_ids.shape
        x = self.emb(text_ids) + self.pos_emb(L)[None, :, :].to(text_ids.device)
        return self.encoder(x)  # [B, L, emb_dim]

# ==============================
#  CROSS-ATTENTION
# ==============================
class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim=None, num_heads=config.num_heads):
        super().__init__()
        context_dim = context_dim or dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(context_dim, dim, bias=False)
        self.to_v = nn.Linear(context_dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, context):
        B, N, C = x.shape
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = q.view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.to_out(out)


# ==============================
#  CONV BLOCKS
# ==============================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,) long (0..T-1)
        half = self.dim // 2
        device = t.device
        emb = torch.exp(-math.log(10000) * torch.arange(half, device=device) / half)
        args = t.float().unsqueeze(1) * emb.unsqueeze(0)  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1))  # (B, dim)
        return emb  # [B, dim]

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )

    def forward(self, x):
        return self.block(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, with_cross_attn=False, time_emb_dim=None):
        super().__init__()
        # используем DoubleConv если он у вас есть (в нём реализована интеграция time_emb)
        self.conv = DoubleConv(in_ch, out_ch, time_emb_dim=time_emb_dim)
        self.downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
        self.with_cross_attn = with_cross_attn
        if with_cross_attn:
            self.cross_attn = CrossAttention(dim=out_ch, context_dim=config.emb_size, num_heads=config.num_heads)
        else:
            self.cross_attn = None

    def forward(self, x, t_emb=None, context=None):
        # x: [B,C,H,W], t_emb: [B, time_emb_dim], context: [B, L, emb_dim] or None
        x = self.conv(x, t_emb)
        if self.cross_attn is not None and context is not None:
            B, C, H, W = x.shape
            x_flat = x.view(B, C, H * W).transpose(1, 2)     # [B, N, C]
            x_flat = self.cross_attn(x_flat, context)        # [B, N, C]
            x = x_flat.transpose(1, 2).view(B, C, H, W)
        x = self.downsample(x)
        return x


# --------------------------
# UpBlock (time-aware + optional cross-attn)
# --------------------------
class UpBlock(nn.Module):
    def __init__(self, x_ch, skip_ch, out_ch, with_cross_attn=False, time_emb_dim=None):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(x_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch, time_emb_dim=time_emb_dim)
        self.with_cross_attn = with_cross_attn
        if with_cross_attn:
            self.cross_attn = CrossAttention(dim=out_ch, context_dim=config.emb_size, num_heads=config.num_heads)
        else:
            self.cross_attn = None

    def forward(self, x, skip, t_emb=None, context=None):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)

        x = self.conv(x, t_emb)

        if self.cross_attn is not None and context is not None:
            B, C, H, W = x.shape
            x_flat = x.view(B, C, H * W).transpose(1, 2)
            x_flat = self.cross_attn(x_flat, context)
            x = x_flat.transpose(1, 2).view(B, C, H, W)

        return x


        

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        # group norm groups heuristic
        gn_groups = min(32, out_channels // 2) if out_channels >= 8 else 1

        # split layers so we can inject time after first conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.gn1 = nn.GroupNorm(gn_groups, out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.gn2 = nn.GroupNorm(gn_groups, out_channels)
        self.act2 = nn.GELU()

        # optional time projection -> out_channels
        if time_emb_dim is not None:
            self.time_proj = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, out_channels)
            )
        else:
            self.time_proj = None

    def forward(self, x, t_emb=None):
        # x: [B, C, H, W], t_emb: [B, time_emb_dim] or None
        x = self.conv1(x)
        if (t_emb is not None) and (self.time_proj is not None):
            # project and add as bias
            tb = self.time_proj(t_emb)            # [B, out_channels]
            x = x + tb.view(x.size(0), -1, 1, 1)
        x = self.gn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.gn2(x)
        x = self.act2(x)
        return x



# ==============================
#  BOTTLENECK + ATTENTION
# ==============================
class TransformerBottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, time_emb_dim=config.time_emb_dim, 
                 num_heads=None, max_img_size=config.img_size):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)           # local conv processing
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_ch),
            nn.GELU(),
            nn.Linear(out_ch, out_ch),
        )

        self.out_ch = out_ch
        self.num_heads = num_heads or config.num_heads

        # positional embeddings for spatial tokens: N = (img_size/8)^2
        factor = 8  # downsample factor (3 stride-2 in your arch)
        spatial = (max_img_size // factor)
        self.max_tokens = spatial * spatial
        # learnable pos emb, shape [1, N, C]
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_tokens, out_ch))

        # Self-attention layer (transformer style)
        self.self_attn = nn.MultiheadAttention(embed_dim=out_ch, num_heads=self.num_heads, batch_first=True)

        # Cross-attention to text
        self.cross_attn = CrossAttention(dim=out_ch, context_dim=emb_dim, num_heads=self.num_heads)

        # FFN (position-wise MLP)
        self.ffn = nn.Sequential(
            nn.Linear(out_ch, out_ch * 4),
            nn.GELU(),
            nn.Linear(out_ch * 4, out_ch),
        )

        # LayerNorms applied on last dim of tokens ([B, N, C])
        self.norm1 = nn.LayerNorm(out_ch)
        self.norm2 = nn.LayerNorm(out_ch)
        self.norm3 = nn.LayerNorm(out_ch)

        # small residual projection if channel mismatch (unlikely here)
        if in_ch != out_ch:
            self.res_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.res_proj = nn.Identity()

        # init pos emb small
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x, text_emb, t_emb):
        B, C, H, W = x.shape
        x = self.conv(x)  

        # add time bias
        time_bias = self.time_mlp(t_emb)     
        x = x + time_bias.view(B, -1, 1, 1)

        # flatten to tokens
        tokens = x.view(B, self.out_ch, H * W).transpose(1, 2)  
        N = tokens.shape[1]
        assert N <= self.max_tokens, f"Too many tokens {N} > {self.max_tokens}"

        pos = self.pos_emb[:, :N, :].expand(B, -1, -1)  # [B, N, C]
        tokens = tokens + pos

        tokens_norm = self.norm1(tokens)
        sa_out, _ = self.self_attn(tokens_norm, tokens_norm, tokens_norm, need_weights=False)
        tokens = tokens + sa_out

        tokens_norm = self.norm2(tokens)
        ca_out = self.cross_attn(tokens_norm, text_emb)
        tokens = tokens + ca_out

        tokens_norm = self.norm3(tokens)
        ffn_out = self.ffn(tokens_norm)
        tokens = tokens + ffn_out

        x = tokens.transpose(1, 2).view(B, self.out_ch, H, W)

        return x


# --------------------------
# UNET (использует config.features и config.cross_attention_levels)
# --------------------------
class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        channels = config.features  # e.g. [64,128,256]
        self.cross_levels = set(config.cross_attention_levels)

        self.downs = nn.ModuleList()
        in_ch = config.in_channels
        for i, out_ch in enumerate(channels):
            with_attn = (i in self.cross_levels)
            self.downs.append(DownBlock(in_ch, out_ch, with_cross_attn=with_attn, time_emb_dim=config.time_emb_dim))
            in_ch = out_ch

        last_feat = config.features[-1]
        self.bottleneck = TransformerBottleneck(in_ch=last_feat, out_ch=last_feat, emb_dim=config.emb_size, max_img_size=config.img_size)


        # ----- постройка ups: прямо len(channels)-1 блоков, чтобы восстановить разрешение -----
        self.ups = nn.ModuleList()
        x_ch = in_ch 

        for j in reversed(range(len(channels))):
            skip_ch = channels[j]
            out_ch = channels[j-1] if (j-1) >= 0 else channels[0]
            with_attn = (j in self.cross_levels)
            self.ups.append(UpBlock(x_ch, skip_ch, out_ch, with_cross_attn=with_attn, time_emb_dim=config.time_emb_dim))
            x_ch = out_ch

        self.final_conv = nn.Conv2d(channels[0], config.out_channels, kernel_size=1)

        self.time_embedding = SinusoidalTimeEmbedding(config.time_emb_dim)
        self.time_embed_net = nn.Sequential(
            nn.Linear(config.time_emb_dim, config.time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(config.time_emb_dim * 4, config.time_emb_dim),
        )

    def forward(self, x, text_emb, t):

        t_emb = self.time_embedding(t)          # [B, time_emb_dim]
        t_emb = self.time_embed_net(t_emb)      # [B, time_emb_dim]

        skips = []
        for down in self.downs:
            x_conv = down.conv(x, t_emb)  

            if getattr(down, "cross_attn", None) is not None and (text_emb is not None):
                B, C, H, W = x_conv.shape
                x_flat = x_conv.view(B, C, H * W).transpose(1, 2) 
                x_flat = down.cross_attn(x_flat, text_emb)
                x_conv = x_flat.transpose(1, 2).view(B, C, H, W)

            skips.append(x_conv)

            x = down.downsample(x_conv)

        x = self.bottleneck(x, text_emb, t_emb)

        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip, t_emb, context=text_emb if up.with_cross_attn else None)
            
        # final projection
        return self.final_conv(x)




# --------------------------
# Model wrapper (Text encoder + UNET)
# --------------------------
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.unet = UNET()

    def forward(self, x, text, t):
        # text: [B, L]
        text_emb = self.text_encoder(text)  # [B, L, emb_dim]
        return self.unet(x, text_emb, t)