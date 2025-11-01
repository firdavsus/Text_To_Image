
import torch
import torch.nn as nn
import math

class Config:
    num_layers=16
    num_heads=24
    emb_size = 768
    image_size = 256
    patches = 16
    grid_size = 16
    ff_hidden_mult = 8
    max_text_len = 256
    max_image_len = 256
    dropout=0.2
    vocab_size = 50000

config = Config()

class Attention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.Q = nn.Linear(in_dim, out_dim, bias=False)
        self.K = nn.Linear(in_dim, out_dim, bias=False)
        self.V = nn.Linear(in_dim, out_dim, bias=False)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        Q = self.Q(x)  # [B, seq, d]
        K = self.K(x)
        V = self.V(x)

        d_k = Q.shape[-1]
        seq_len = Q.size(1)

        # Q @ Kᵀ  (ты использовал Q@Qᵀ — это баг, мы это исправляем)
        attention_score = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)  # [B, seq, seq]

        if mask is not None:
            # Приводим mask к [B, 1, seq, seq] или хотя бы к совместимому размеру
            if mask.dim() == 4:
                mask = mask[:, :, :seq_len, :seq_len]
                # Сжимаем для совместимости: [B,1,seq,seq] -> [B,seq,seq]
                mask = mask.squeeze(1).squeeze(1)
            mask = mask.bool() 
            attention_score = attention_score.masked_fill(~mask, float('-1e-6'))

        probs = self.softmax(attention_score)
        output = probs @ V  # [B, seq, d]

        return output

class MLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_dim = config.emb_size
        self.heads = config.num_heads

        assert self.in_dim % self.heads == 0

        self.for_each_head = self.in_dim // self.heads

        self.attn = nn.ModuleList(
            [Attention(self.for_each_head, self.for_each_head) for _ in range(self.heads)]
        )

        self.ml_head = nn.Linear(self.in_dim, self.in_dim)

    def forward(self, embedding, mask=None):

        batch, seq_len, d_emb = embedding.shape
        
         # (1 x 64 x 512) --> (1 x 64 x 8 x 64) which means : 
        # (Batch, seq_len,d_emb) -->(Batch, seq_len, heads, attn_dim)  
        mha_input = embedding.view(batch, seq_len, self.heads, self.for_each_head)

        # (batch, seq_len, n_heads, attn_dim) -> (batch, n_heads, seq_len, attn_dim)    
        mha_input = mha_input.permute(0, 2, 1 ,3) # it divides the embedding to heads
        if mask is not None:
            # Ensure [B, seq_len, seq_len] per head
            mask = mask.squeeze(1)  # [1, seq, seq] -> [seq, seq]
            mask = mask.expand(batch, seq_len, seq_len)  # [B, seq, seq]
        
        outs = [head(mha_input[:, i, :, :], mask=mask) for i, head in enumerate(self.attn)]
        out = torch.cat(outs, dim=-1)
        
        out = self.ml_head(out)
        
        return out
    

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = MLA()
        self.norm1 = nn.RMSNorm(config.emb_size)
        self.norm2 = nn.RMSNorm(config.emb_size)

        hidden_dim = config.emb_size * config.ff_hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(config.emb_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.emb_size),
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        attn_out = self.attention(x, mask=mask) 
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x
    
class ConvEncoder(nn.Module):
    def __init__(self, embedding_dim=config.emb_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # 256 -> 128
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(128, embedding_dim, 4, 2, 1), # 32 -> 16
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)  
    
class ImagePositionalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.row_embed = nn.Embedding(config.image_size, config.emb_size)
        self.col_embed = nn.Embedding(config.image_size, config.emb_size)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        device = x.device
        
        # row and column indices
        rows = torch.arange(H, device=device)
        cols = torch.arange(W, device=device)
        
        # embeddings: [H, emb_size] and [W, emb_size]
        row_emb = self.row_embed(rows)[:, None, :]  # [H,1,emb_size]
        col_emb = self.col_embed(cols)[None, :, :]  # [1,W,emb_size]
        
        # broadcast to [H,W,emb_size] and then add
        pos_emb = row_emb + col_emb  # [H, W, emb_size]
        pos_emb = pos_emb.permute(2,0,1).unsqueeze(0).repeat(B,1,1,1)
        return pos_emb

def make_causal_cond_mask(T_text, T_image, device):
    # full mask shape: [seq_len, seq_len] bool where True = allowed
    seq_len = T_text + T_image
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    # text rows: text can see all tokens (text + image)
    mask[:T_text, :] = True

    # image rows: can see all text tokens
    mask[T_text:, :T_text] = True
    # image rows among themselves: causal (lower triangular)
    img_causal = torch.tril(torch.ones(T_image, T_image, dtype=torch.bool, device=device))
    mask[T_text:, T_text:] = img_causal
    # expand to batch/head shape later as needed
    return mask  # shape [seq_len, seq_len]

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        # text part
        self.text_emb = nn.Embedding(config.vocab_size, config.emb_size)
        self.text_pos = nn.Embedding(config.max_text_len, config.emb_size)

        # image embedder
        self.image_emb = ConvEncoder(config.emb_size)
        self.pos_emb_layer = ImagePositionalEmbedding()

        self.text_blocks = nn.ModuleList([Block() for _ in range(config.num_layers)])
        self.norm = nn.RMSNorm(config.emb_size)

        self.lm_head = nn.Linear(config.emb_size, (config.patches**2) * 3)

    def forward(self, text, images=None):
        B, T1 = text.shape
        if images is not None:
            B, C, H, W = images.shape
    
            # Image features -> tokens
            img_feat = self.image_emb(images)                   # [B, emb, 16, 16]
            pos_emb = self.pos_emb_layer(img_feat)
            img_feat = img_feat + pos_emb
            image_emb = img_feat.flatten(2).transpose(1, 2)     # [B, 256, emb] if 16x16
    
            T2 = image_emb.shape[1]  # <-- REAL image token count!
    
            # ✅ Build mask for (B, heads, seq_len, seq_len)
            mask = make_causal_cond_mask(T1, T2, text.device)
            mask = mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq, seq]
    
            # Embeddings
            text_positions = torch.arange(T1, device=text.device)
            text_emb = self.text_emb(text) + self.text_pos(text_positions)
    
            seq = torch.cat([text_emb, image_emb], dim=1)
    
            for block in self.text_blocks:
                seq = block(seq, mask=mask)
    
            seq = self.norm(seq)
            image_tokens = seq[:, T1:, :]
            image_preds = self.lm_head(image_tokens)
            image_preds = image_preds.view(B, T2, 3, 16, 16)
    
            return image_preds
        else:
            H = W = config.grid_size *  config.patches
            T2 = config.grid_size**2
    
            # Initialize dummy image tokens with zeros
            image_emb = torch.zeros(B, T2,  config.emb_size, device=text.device)
            mask = make_causal_cond_mask(T1, T2, text.device)
            mask = mask.unsqueeze(0).unsqueeze(1)
    
            text_positions = torch.arange(T1, device=text.device)
            text_emb = self.text_emb(text) + self.text_pos(text_positions)
    
            seq = torch.cat([text_emb, image_emb], dim=1)
    
            # Pass through transformer blocks autoregressively
            for block in self.text_blocks:
                seq = block(seq, mask=mask)
            seq = self.norm(seq)
            image_tokens = seq[:, T1:, :]
            image_preds = self.lm_head(image_tokens)
            # reshape to full image
            B, T2, C = image_preds.shape[0], image_preds.shape[1], image_preds.shape[2]
            image_preds = image_preds.view(B, 3, H, W)  # flatten patches back to full image
    
            return image_preds