import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from transformers import BertModel, BertTokenizer

class Config:
    features = [64, 128, 256, 512]
    cross_attn = [2, 3]   
    in_channels = 4           
    out_channels = 4
    image_size = 4
    time_dim = 768
    text_emb_dim = 768
    vocab_size = 50256
    dropout = 0.1
    num_heads = 8
    num_layers = 6
    max_len = 77
    model_bert = "distilbert-base-uncased"

config = Config()

# ---------------- time embedding ----------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        inv_freq = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        pos = t[:, None].float() * inv_freq[None, :]     
        emb = torch.cat([torch.sin(pos), torch.cos(pos)], dim=-1)
        return emb

# ---------------- time-aware conv block ----------------
class TimeAwareDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        groups = 8 if out_ch % 8 == 0 else 1
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act2 = nn.SiLU()

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, out_ch),
            nn.SiLU()
        )

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)

        t = self.time_mlp(t_emb)[:, :, None, None]  
        h = h + t

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        return h

# ---------------- cross-attention ----------------
class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim, num_heads=config.num_heads):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(context_dim, dim)
        self.to_v = nn.Linear(context_dim, dim)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, context):
        B, N, C = x.shape
        q = self.to_q(x).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)  
        k = self.to_k(context).view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.to_v(context).view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale   
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)                            
        out = out.transpose(1, 2).contiguous().view(B, N, C)    
        return self.to_out(out)

# ---------------- text encoder ----------------
# class TextEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.emb = nn.Embedding(config.vocab_size, config.text_emb_dim)
#         self.pos_emb = nn.Embedding(config.max_len, config.text_emb_dim)
#         layer = nn.TransformerEncoderLayer(
#             d_model=config.text_emb_dim,
#             nhead=config.num_heads,
#             dim_feedforward=config.text_emb_dim * 4,
#             dropout=config.dropout,
#             activation="gelu",
#             batch_first=True,
#         )
#         self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)

#     def forward(self, x):
#         B, L = x.shape
#         device = x.device
#         pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L) 
#         x = self.emb(x) + self.pos_emb(pos_ids)
#         x = self.encoder(x)  
#         return x

bert_name = config.model_bert
bert_tok = BertTokenizer.from_pretrained(bert_name)
# ensure pad token exists
if bert_tok.pad_token is None:
    bert_tok.add_special_tokens({"pad_token": "[PAD]"})
class TextEncoder(nn.Module):
    def __init__(self, pretrained=bert_name, freeze=True, proj_dim=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained, return_dict=True)

        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

        bert_hidden = self.bert.config.hidden_size
        self.proj = None
        if proj_dim is not None and proj_dim != bert_hidden:
            self.proj = nn.Linear(bert_hidden, proj_dim)

    def forward(self, x):
        pad_id = bert_tok.pad_token_id
        attn_mask = (x != pad_id).long()

        out = self.bert(input_ids=x, attention_mask=attn_mask)
        last = out.last_hidden_state         
        if self.proj is not None:
            last = self.proj(last)
        return last

# ---------------- UNET ----------------
class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        feats = config.features
        time_dim = config.time_dim
        text_dim = config.text_emb_dim

        # lists
        self.downs = nn.ModuleList()
        self.down_cross = nn.ModuleList()
        self.ups_convT = nn.ModuleList()
        self.ups_double = nn.ModuleList()
        self.up_cross = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        # time embed
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        self.text_enc = TextEncoder()

        in_ch = config.in_channels

        for lvl, feat in enumerate(feats):
            self.downs.append(TimeAwareDoubleConv(in_ch, feat, time_dim))
            if lvl in config.cross_attn:
                self.down_cross.append(CrossAttention(feat, text_dim, num_heads=config.num_heads))
            else:
                self.down_cross.append(nn.Identity())
            in_ch = feat

        self.bottleneck = TimeAwareDoubleConv(feats[-1], feats[-1] * 2, time_dim)

        for idx_rev, feat in enumerate(reversed(feats)):
            orig_level = len(feats) - 1 - idx_rev

            self.ups_convT.append(nn.ConvTranspose2d(feat * 2, feat, kernel_size=2, stride=2))

            self.ups_double.append(TimeAwareDoubleConv(feat * 2, feat, time_dim))
            if orig_level in config.cross_attn:
                self.up_cross.append(CrossAttention(feat, text_dim, num_heads=config.num_heads))
            else:
                self.up_cross.append(nn.Identity())

        self.final_conv = nn.Conv2d(feats[0], config.out_channels, kernel_size=1)

    def forward(self, x, t, text):

        B = x.size(0)
        t_emb = self.time_emb(t)          
        t_emb = self.time_mlp(t_emb)     

        text_emb = self.text_enc(text)   

        skips = []
        for i, down in enumerate(self.downs):
            x = down(x, t_emb)
            cross = self.down_cross[i]
            if not isinstance(cross, nn.Identity):
                B_, C, H, W = x.shape
                x_flat = x.view(B_, C, H * W).transpose(1, 2)  
                x_flat = cross(x_flat, text_emb)          
                x = x_flat.transpose(1, 2).view(B_, C, H, W)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x, t_emb)

        for idx in range(len(self.ups_convT)):
            x = self.ups_convT[idx](x)
            skip = skips[-1 - idx]
            if x.shape != skip.shape:
                x = TF.resize(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups_double[idx](x, t_emb)

            cross = self.up_cross[idx]
            if not isinstance(cross, nn.Identity):
                B_, C, H, W = x.shape
                x_flat = x.view(B_, C, H * W).transpose(1, 2)
                x_flat = cross(x_flat, text_emb)
                x = x_flat.transpose(1, 2).view(B_, C, H, W)

        return self.final_conv(x)


    
def test():
    m = UNET().cuda()
    batch = 2
    H = W = 256
    x = torch.randn(batch, config.in_channels, H, W).cuda()
    t = torch.randint(0, 1000, (batch,), device='cuda')
    tokens = torch.randint(0, config.vocab_size, (batch, config.max_len)).cuda()

    out = m(x, t, tokens)
    print("out shape:", out.shape)

if __name__ == "__main__":
    test()