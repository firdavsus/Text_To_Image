import torch
from transformers import AutoTokenizer
from model import DiffusionModel
from ema import EMA
from UNET import config
import torchvision.transforms.functional as TF
import os

import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image

def _detect_max_len_from_text_encoder(model, default=128):
    # Попытаться извлечь длину позиционных эмбеддингов (если есть)
    try:
        pos = model.text_encoder.pos_emb  # предположение о названии
        L = pos.shape[0]
        return int(L)
    except Exception:
        return default

@torch.no_grad()
def diffusion_evolution(model, tokenizer, prompt="a cat", steps=500, save_every=50, device="cuda", output_dir="evolution"):
    os.makedirs(output_dir, exist_ok=True)
    model = model.to(device).eval()

    # Tokenize text
    MAX_LEN = _detect_max_len_from_text_encoder(model, default=128)
    toks = tokenizer(prompt, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")["input_ids"].to(device)
    text_emb = model.text_encoder(toks).float()

    # Init latent
    B = 1
    latent_shape = (B, getattr(model.vae, "latent_channels", 4), 32, 32)
    z = torch.randn(latent_shape, device=device) * getattr(model, "vae_latent_scale", 0.18215)

    # Get diffusion params
    betas = model.betas.to(device).float()
    alphas = model.alphas.to(device).float()
    alpha_bars = model.alpha_bars.to(device).float()

    timesteps = list(range(steps - 1, -1, -1))

    saved_paths = []
    for idx, t in enumerate(timesteps):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

        # Predict noise
        eps = model.unet(z, text_emb, t_tensor)

        # DDPM update
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]
        beta_t = betas[t]

        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)

        z = coef1 * (z - coef2 * eps)

        if t > 0:
            z = z + torch.sqrt(beta_t) * torch.randn_like(z)

        # Decode and save every N steps
        if idx % save_every == 0 or t == 0:
            img = model.vae.decoder(z).clamp(-1, 1)
            img = (img[0] * 0.5 + 0.5).detach().cpu()  # [-1,1] → [0,1]
            img = TF.to_pil_image(img)
            path = os.path.join(output_dir, f"step_{t:04d}.png")
            img.save(path)
            saved_paths.append(path)
            print(f"Saved: {path}")

    return saved_paths



# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 500

# tokenizer
tok = AutoTokenizer.from_pretrained("gpt2")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

config.vocab_size = tok.vocab_size

def tokenize_batch(texts):
    out = tok(texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    return out["input_ids"].to(device)

# model 
betas = torch.linspace(1e-4, 2e-2, T).to(device)

net = DiffusionModel(betas).to(device)

ema = EMA(net, decay=0.9999, device=None)

checkpoint = torch.load(f"save/model-208000.pt", map_location=device)
net.load_state_dict(checkpoint["model"])
ema.shadow = checkpoint["ema"] 
net.eval()

def get_image(prompt, steps=500, guidance=1.5, save_every=50):
    with torch.no_grad():
        ema.apply_shadow(net)

        # run evolution sample (returns list of saved image paths)
        saved_paths = diffusion_evolution(
            model=net,
            tokenizer=tok,
            prompt=prompt,
            steps=steps,
            save_every=save_every,
            device="cuda",
            output_dir="output_evolution"
        )

        # restore EMA
        ema.restore(net)

        print(f"\n✅ Done! Saved {len(saved_paths)} evolution frames to output_evolution/\n")
        return saved_paths


if __name__ == "__main__":
    while True:
        prompt = input("Prompt: ")
        get_image(prompt)
        print("Done")