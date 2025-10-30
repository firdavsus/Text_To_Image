import io, os, math, time, requests, random
from PIL import Image
import torch.nn.functional as F
from tqdm.auto import tqdm
import torch, torch.nn as nn, torch.optim as optim
import torchvision.transforms.functional as TF
from datasets import load_dataset
from transformers import AutoTokenizer
from itertools import islice
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from model import DiffusionModel
from ema import EMA
from UNET import config
from torch.utils.data import Dataset
from torchvision import transforms
import os
import json
from PIL import Image
from torch.utils.data import DataLoader

# ----------------- CONFIG -----------------
class CFG:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 1e-4
    T = 500
    log_every = 100
    save_every = 1000
    batch_size = 8
    out_dir = "samples"
    max_len = config.max_len
    img_size = config.img_size
    dataset = "conceptual_captions"
    accum_steps=4
    epoch=10

cfg = CFG()
os.makedirs(cfg.out_dir, exist_ok=True)

# ----------------- tokenizer (BPE: GPT-2) -----------------
tok = AutoTokenizer.from_pretrained("gpt2")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

config.vocab_size = tok.vocab_size

def tokenize_batch(texts):
    out = tok(texts, padding="max_length", truncation=True, max_length=cfg.max_len, return_tensors="pt")
    return out["input_ids"].to(cfg.device)

# ----------------- small DDPM schedule -----------------
betas = torch.linspace(1e-4, 2e-2, cfg.T).to(cfg.device)
alphas = 1 - betas
alphas_cum = torch.cumprod(alphas, dim=0)
sqrt_ac = torch.sqrt(alphas_cum); sqrt_1mac = torch.sqrt(1 - alphas_cum)
def q_sample(x0, t, noise=None):
    if noise is None: noise = torch.randn_like(x0)
    a = sqrt_ac[t].view(-1,1,1,1); b = sqrt_1mac[t].view(-1,1,1,1)
    return a * x0 + b * noise

# ----------------- sampling (simple DDPM reverse) -----------------
@torch.no_grad()
def sample_model(model, toks, steps=None, guidance=1.0, device="cuda"):
    model = model.to(device)
    B = toks.size(0)
    toks = toks.to(device)

    latent_shape = (B, 4, 32, 32)
    z = torch.randn(latent_shape, device=device)  # standard Gaussian
    z *= 0.18215  # scale matches training latent

    if steps is None:
        steps = model.betas.size(0)

    betas = model.betas
    alphas = model.alphas
    alpha_bars = model.alpha_bars

    text_emb = model.text_encoder(toks).float()
    if guidance != 1.0:
        empty_toks = torch.full_like(toks, fill_value=tok.pad_token_id)
        text_emb_uncond = model.text_encoder(empty_toks).float()

    timesteps = torch.arange(steps-1, -1, -1, device=device)

    for t in timesteps:
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        eps = model.unet(z, text_emb, t_batch)
        if guidance != 1.0:
            eps_uncond = model.unet(z, text_emb_uncond, t_batch)
            eps = eps_uncond + guidance * (eps - eps_uncond)

        alpha = alphas[t]
        alpha_bar = alpha_bars[t]
        beta = betas[t]

        z = (1 / torch.sqrt(alpha)) * (z - (beta / torch.sqrt(1 - alpha_bar)) * eps)
        if t > 0:
            z += torch.sqrt(beta) * torch.randn_like(z)

    # Decode latent
    img = model.vae.decoder(z)
    return img.clamp(-1, 1)


# ----------------- model + optimizer -----------------
net = DiffusionModel(betas).to(cfg.device)
state = torch.load("checkpoint/old-2.pth", map_location=cfg.device)
from collections import OrderedDict

new_state = OrderedDict()
for k, v in state.items():
    new_state[k.replace("module.", "")] = v

net.vae.load_state_dict(new_state, strict=True)

# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs")
#     net = torch.nn.DataParallel(net)

decay, no_decay = [], []
for n, p in net.named_parameters():
    if not p.requires_grad:
        continue
    if n.endswith(".bias") or "norm" in n.lower() or "groupnorm" in n.lower() or "embedding" in n.lower():
        no_decay.append(p)
    else:
        decay.append(p)

opt = torch.optim.AdamW([
    {"params": decay, "weight_decay": 1e-2},
    {"params": no_decay, "weight_decay": 0.0},
], lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8)

mse = nn.MSELoss()
ema = EMA(net, decay=0.9999, device=None)


# ----------------- stream dataset iterator (light heuristic) -----------------
class TextImageDataset(Dataset):
    def __init__(self, images_dir, json_path, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        with open(json_path, "r", encoding="utf-8") as f:
            self.descriptions = json.load(f)

        # Берём только изображения с поддерживаемыми расширениями
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        # Ключ в captions.json соответствует имени файла без расширения
        key = os.path.splitext(image_name)[0]
        description = self.descriptions.get(key, "")
        return image, description

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = TextImageDataset("all_images", "captions.json", transform=transform)
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

# ----------------- training loop using chunked download -----------------
pbar = tqdm(total=len(dataset))
step = 0
net.train()

losses = []
scaler = torch.cuda.amp.GradScaler()  # initialize once before training

for epoch in range(cfg.epoch):
    print(f"=== Epoch {epoch+1}/{cfg.epoch} ===")
    for i, (images, captions) in enumerate(dataloader):
        x0 = images.to(cfg.device)
        toks = tokenize_batch(captions)

        t = torch.randint(0, cfg.T, (x0.size(0),), device=cfg.device).long()

        with torch.cuda.amp.autocast(dtype=torch.float16):
            pred_eps, target_eps = net(x0, toks, t)

            loss = mse(pred_eps, target_eps) / cfg.accum_steps

        # Backward
        scaler.scale(loss).backward()

        if (step + 1) % cfg.accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            ema.update(net)

        step += 1

        # Logging
        if step % cfg.log_every == 0:
            true_loss = loss.item() * cfg.accum_steps
            print(f"[step {step}] loss={true_loss:.4f}")
            losses.append((step, true_loss))

            # optional: preview mean/std
            with torch.no_grad():
                print("target_eps mean/std:", target_eps.mean().item(), target_eps.std().item())
                print("pred_eps mean/std:", pred_eps.mean().item(), pred_eps.std().item())

        # Sampling checkpoint
        if step % cfg.save_every == 0:
            net.eval()
            with torch.no_grad():
                prompts = "a cat portrait"
                toks_s = tokenize_batch(prompts)
                ema.apply_shadow(net)
                sampled = sample_model(net, toks_s, guidance=1.5)
                ema.restore(net)
                imgs_out = (sampled.clamp(-1,1)+1)/2
                for j, img in enumerate(imgs_out):
                    TF.to_pil_image(img.cpu()).save(os.path.join(cfg.out_dir, f"step{step}_samp{j}.png"))
            net.train()

            torch.save({
                "model": net.state_dict(),
                "ema": ema.state_dict(),
            }, f"save/model-{step}.pt")

pbar.close()
print("Training Done.")