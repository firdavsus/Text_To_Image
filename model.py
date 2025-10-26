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
    emb_size = 128
    max_len = 256
    num_heads = 4
    num_layers = 4
    dropout = 0.1
    in_channels = 3
    out_channels = 3
    img_size = 128
    time_emb_dim = 128
    features = [64, 128, 256]


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
    def __init__(self, dim, context_dim=None, num_heads=4):
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


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # group norm with groups = min(32, out_channels//2) typical
        gn_groups = min(32, out_channels // 2) if out_channels >= 8 else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# ==============================
#  BOTTLENECK + ATTENTION
# ==============================
class BottleneckWithAttention(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, time_emb_dim=config.time_emb_dim):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.attn = CrossAttention(dim=out_ch, context_dim=emb_dim, num_heads=4)
        # project time embedding to out_ch and add as bias
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_ch),
            nn.GELU(),
            nn.Linear(out_ch, out_ch),
        )

    def forward(self, x, text_emb, t_emb):
        # t_emb: [B, time_emb_dim]
        B, _, _, _ = x.shape
        x = self.conv(x)                       # [B,C,H,W]
        time_bias = self.time_mlp(t_emb)       # [B, C]
        # add time to features (broadcast)
        x = x + time_bias.view(B, -1, 1, 1)
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).transpose(1, 2)  # [B, N, C]
        x_flat = self.attn(x_flat, text_emb)         # text_emb: [B, L, emb_dim]
        x = x_flat.transpose(1, 2).view(B, C, H, W)
        return x


# ==============================
#  U-NET
# ==============================
class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        in_channels = config.in_channels

        for feature in config.features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(config.features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.time_embed_net = nn.Sequential(
            nn.Linear(config.time_emb_dim, config.time_emb_dim*4),
            nn.GELU(),
            nn.Linear(config.time_emb_dim*4, config.time_emb_dim),
        )
        self.time_embedding = SinusoidalTimeEmbedding(config.time_emb_dim)
        self.bottleneck = BottleneckWithAttention(
            config.features[-1], config.features[-1] * 2, config.emb_size
        )
        self.final_conv = nn.Conv2d(config.features[0], config.out_channels, kernel_size=1)

    def forward(self, x, text_emb, t):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        t_emb = self.time_embedding(t)            # [B, time_emb_dim]
        t_emb = self.time_embed_net(t_emb)
        x = self.bottleneck(x, text_emb, t_emb)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

# ==============================
#  MODEL
# ==============================
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNET()
        self.text_encoder = TextEncoder()

    def forward(self, x, text, t):
        text_emb = self.text_encoder(text)
        return self.unet(x, text_emb, t)
    