# minimal_laion_train.py
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
from model import Model, config
from ema import EMA

# ----------------- CONFIG -----------------
class CFG:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = config.img_size
    batch = 1
    steps = 10000
    lr = 1e-4
    T = 1000
    log_every = 25
    save_every = 100
    out_dir = "samples"
    max_len = config.max_len
    dataset = "conceptual_captions"
    chunk_size = 1000
    accum_steps=1
    epoch = 30

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

# ----------------- simple image fetch+prep -----------------
def fetch_image(url, size=cfg.img_size, timeout=4):
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent":"curl/7.64"})
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        # center-crop square then resize
        W,H = im.size; m = min(W,H)
        im = im.crop(((W-m)//2, (H-m)//2, (W+m)//2, (H+m)//2)).resize((size,size), Image.BICUBIC)
        t = TF.to_tensor(im) * 2 - 1 
        return t
    except Exception:
        return None

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
def sample_model(model, toks, shape, guidance=1.0):
    B = shape[0]
    x = torch.randn(shape, device=cfg.device)
    for i in reversed(range(cfg.T)):
        t = torch.full((B,), i, device=cfg.device, dtype=torch.long)

        if guidance != 1.0:
            # classifier-free guidance: concat null + conditioned
            null = torch.full_like(toks, tok.pad_token_id) 
            xin = torch.cat([x, x], dim=0)            # [2B, C, H, W]
            tin = torch.cat([null, toks], dim=0)      # [2B, L]
            tinc = torch.full((B*2,), i, device=cfg.device, dtype=torch.long)  # repeat t
            eps = model(xin, tin, tinc)               # now pass t
            eps = eps.chunk(2, 0)
            eps = eps[0] + guidance * (eps[1] - eps[0])
        else:
            # no guidance
            eps = model(x, toks, t)                   # pass t here too

        beta = betas[i]; a = alphas[i]; ac = alphas_cum[i]
        z = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
        # DDPM reverse update (same as you had)
        x = (1.0 / math.sqrt(a)) * (x - (beta / math.sqrt(1.0 - ac)) * eps) + math.sqrt(beta) * z

    return x


# ----------------- model + optimizer -----------------
net = Model().to(cfg.device)
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Apply to model
net.apply(init_weights)
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
dataset = load_dataset(cfg.dataset, split="train")
cfg.steps=len(dataset)//cfg.batch
def iter_pairs_chunked(dataset, max_workers=8, chunk_size=cfg.chunk_size):
    """
    Concurrent image downloader that preloads images in chunks.
    Uses a persistent ThreadPoolExecutor for efficiency.
    """
    def fetch_worker(e):
        img = fetch_image(e["image_url"])  # must return a tensor or None
        return img, e["caption"]

    dataset_iter = iter(dataset)
    total_downloaded = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            batch_ex = list(islice(dataset_iter, chunk_size))
            if not batch_ex:
                break

            futures = [executor.submit(fetch_worker, ex) for ex in batch_ex]
            preloaded_imgs, preloaded_texts = [], []

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Chunk loading"):
                try:
                    img, cap = fut.result()
                    if img is not None:
                        preloaded_imgs.append(img)
                        preloaded_texts.append(cap)
                        total_downloaded += 1
                except Exception as e:
                    print("⚠️ fetch failed:", e)

            yield preloaded_imgs, preloaded_texts, total_downloaded


# ----------------- training loop using chunked download -----------------
pbar = tqdm(total=cfg.steps)
step = 0
net.train()

losses = []
scaler = torch.cuda.amp.GradScaler()  # initialize once before training

for epoch in range(cfg.epoch):
    print(f"=== Epoch {epoch+1}/{cfg.epoch} ===")
    for imgs_chunk, texts_chunk, downloaded_so_far in iter_pairs_chunked(dataset):
        print(f"Total images downloaded so far: {downloaded_so_far}")

        for i in range(0, len(imgs_chunk), cfg.batch):
            x0 = torch.stack(imgs_chunk[i:i+cfg.batch], dim=0).to(cfg.device)
            toks = tokenize_batch(texts_chunk[i:i+cfg.batch])

            t = torch.randint(0, cfg.T, (x0.size(0),), device=cfg.device).long()
            noise = torch.randn_like(x0)
            xt = q_sample(x0, t, noise=noise)

            # ✅ use autocast for forward + loss in FP16
            with torch.cuda.amp.autocast(dtype=torch.float16):
                pred = net(xt, toks, t)
                loss = mse(pred, noise) / cfg.accum_steps

            # ✅ scale the loss before backward
            scaler.scale(loss).backward()

            if (step + 1) % cfg.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

                # ✅ unscale + optimizer step safely
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

                ema.update(net)

            step += 1
            pbar.update(1)

            # --- logging and preview as before ---
            if step % cfg.log_every == 0:
                true_loss = loss.item() * cfg.accum_steps
                print(f"[step {step}] loss={true_loss:.4f}")
                losses.append((step, true_loss))

                with torch.no_grad():
                    print("noise mean/std:", noise.mean().item(), noise.std().item())
                    print("pred mean/std:", pred.mean().item(), pred.std().item())
                    print("batch mse:", F.mse_loss(pred, noise).item(), "baseline zero-predict:", (noise**2).mean().item())

                plt.figure(figsize=(6,4))
                steps_arr, loss_arr = zip(*losses)
                plt.plot(steps_arr, loss_arr, marker='o')
                plt.xlabel("Step")
                plt.ylabel("Loss")
                plt.title("Training Loss Curve")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(cfg.out_dir, "loss_curve.png"))
                plt.close()

            if step % cfg.save_every == 0:
                net.eval()
                with torch.no_grad():
                    prompts = ["a red car", "a cat portrait", "a person standing in front of a building"]
                    toks_s = tokenize_batch(prompts)
                    ema.apply_shadow(net)
                    sampled = sample_model(net, toks_s, (len(prompts), 3, cfg.img_size, cfg.img_size), guidance=1.5)
                    ema.restore(net)
                    imgs_out = (sampled.clamp(-1,1)+1)/2
                    for j, img in enumerate(imgs_out):
                        TF.to_pil_image(img.cpu()).save(os.path.join(cfg.out_dir, f"step{step}_samp{j}.png"))
                net.train()

                torch.save({
                    "model": net.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "step": step
                }, f"save/model-{step}.pt")


pbar.close()
print("Training Done.")