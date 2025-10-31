from VAE import VAE
from new_UNET import UNET

# # class Model(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.text_encoder = TextEncoder()
# #         self.unet = UNET()
# #         self.vae = VAE()
        
# #         self.vae.eval()
# #         for param in self.vae.parameters():
# #             param.requires_grad = False

# #     def forward(self, x, text, t):
# #         with torch.no_grad():
# #             z = self.vae.encoder(img)
        
# #         text_emb = self.text_encoder(text)  

# #         unet_work = self.unet(z, text_emb, t)

# #         with torch.no_grad():
# #             out = self.vae.decoder(unet_work)  
# #         return out

import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self, betas):
        super().__init__()
        self.vae = VAE()
        self.unet = UNET()
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1 - betas)
        self.register_buffer("alpha_bars", torch.cumprod(1 - betas, dim=0))

        # Freeze VAE
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

    def forward(self, images, text, t):
        with torch.no_grad():
            z = self.vae.encoder(images) 

        # 2️⃣ Sample noise
        eps = torch.randn_like(z)

        # Get corresponding alpha_bar for each t
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)  # [B,1,1,1]

        # 3️⃣ Add noise to latent
        z_t = torch.sqrt(alpha_bar_t) * z + torch.sqrt(1 - alpha_bar_t) * eps

        # 5️⃣ UNet predicts the noise
        pred_eps = self.unet(z_t, t, text)

        # Return prediction and target noise for loss
        return pred_eps, eps


# from itertools import islice
# import os
# import json
# import asyncio
# from io import BytesIO
# from PIL import Image
# from datasets import load_dataset
# from tqdm.asyncio import tqdm
# import aiohttp
# import aiofiles
# import async_timeout

# # ---------------- CONFIG ----------------
# output_folder = "all_images"
# captions_file = "captions.json"
# os.makedirs(output_folder, exist_ok=True)

# dataset_split = "train"      # or "validation" / "test"
# max_concurrent = 64          # max simultaneous connections
# chunk_size = 1024             # process dataset in chunks
# timeout = 4                   # seconds per request
# img_size = 256
# save_every = 25            # save captions periodically

# # ---------------- helpers ----------------
# async def fetch_image_aio(session, url, size=img_size, timeout=timeout):
#     """Download and center-crop+resize image asynchronously."""
#     if not url:
#         return None
#     try:
#         async with async_timeout.timeout(timeout):
#             async with session.get(url, headers={"User-Agent":"Mozilla/5.0"}) as resp:
#                 if resp.status != 200:
#                     return None
#                 content = await resp.read()
#                 im = Image.open(BytesIO(content)).convert("RGB")
#                 W, H = im.size
#                 m = min(W, H)
#                 im = im.crop(((W - m)//2, (H - m)//2, (W + m)//2, (H + m)//2)).resize((size, size), Image.BICUBIC)
#                 return im
#     except Exception:
#         return None

# async def download_worker(session, idx, ex):
#     """Download single image and return (idx, caption) or None."""
#     url = ex.get("image_url")
#     caption = ex.get("caption", "")
#     if not url:
#         return None
#     img = await fetch_image_aio(session, url)
#     if img is None:
#         return None
#     tmp_path = os.path.join(output_folder, f"{idx}.tmp")
#     final_path = os.path.join(output_folder, f"{idx}.jpg")
#     try:
#         img.save(tmp_path, format="JPEG", quality=92)
#         os.replace(tmp_path, final_path)
#         return idx, caption
#     except Exception:
#         if os.path.exists(tmp_path):
#             os.remove(tmp_path)
#         return None

# def chunked_iterable(it, size):
#     it = iter(it)
#     while True:
#         chunk = list(islice(it, size))
#         if not chunk:
#             break
#         yield chunk

# # ---------------- resume support ----------------
# captions = {}
# downloaded_ids = set()
# if os.path.exists(captions_file):
#     try:
#         with open(captions_file, "r", encoding="utf-8") as f:
#             captions = json.load(f)
#         downloaded_ids = set(int(k) for k in captions.keys())
#         print(f"Resuming: loaded {len(captions)} captions")
#     except Exception:
#         captions = {}
#         downloaded_ids = set()

# # ---------------- main async downloader ----------------
# dataset = load_dataset("conceptual_captions", split=dataset_split)
# dataset_iter = iter(dataset)
# try:
#     total_estimate = len(dataset)
# except Exception:
#     total_estimate = None

# async def main():
#     global captions, downloaded_ids
#     global_idx = 0
#     pbar = tqdm(total=total_estimate, desc="Attempted", unit="img")
#     pbar.set_postfix(saved=len(captions), refresh=False)

#     connector = aiohttp.TCPConnector(limit=max_concurrent, ssl=False)
#     timeout_obj = aiohttp.ClientTimeout(total=None)
#     async with aiohttp.ClientSession(connector=connector, timeout=timeout_obj) as session:
#         for chunk in chunked_iterable(dataset_iter, chunk_size):
#             tasks = []
#             for i, ex in enumerate(chunk):
#                 idx = global_idx + i
#                 if idx in downloaded_ids:
#                     pbar.update(1)
#                     continue
#                 tasks.append(download_worker(session, idx, ex))

#             for fut in asyncio.as_completed(tasks):
#                 res = await fut
#                 pbar.update(1)
#                 if res is not None:
#                     idx_saved, caption = res
#                     captions[str(idx_saved)] = caption
#                     downloaded_ids.add(idx_saved)
#                 if len(captions) % 100 == 0:
#                     pbar.set_postfix(saved=len(captions), refresh=True)

#             global_idx += len(chunk)

#             # periodically save captions
#             if len(captions) > 0 and len(captions) % save_every == 0:
#                 print("it is zsaved")
#                 async with aiofiles.open(captions_file, "w", encoding="utf-8") as f:
#                     await f.write(json.dumps(captions, ensure_ascii=False))
#                 pbar.set_postfix(saved=len(captions), refresh=True)

#     # final save
#     if len(captions) > 0:
#         async with aiofiles.open(captions_file, "w", encoding="utf-8") as f:
#             await f.write(json.dumps(captions, ensure_ascii=False))
#     pbar.close()
#     print("Done. saved images with captions:", len(captions))

# # ---------------- run ----------------
# asyncio.run(main())

