from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torch
import os
import json
import lpips
import torch.nn.functional as F
from tqdm import tqdm

from model import Transformer, config

#### some configs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
accum_steps = 1
num_epochs = 30
learning_rate = 2e-4
save_each = 500
#### end of params


#### tokenizer
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
config.vocab_size = len(tokenizer)

def tokenize_texts(texts, max_len=config.max_text_len, device="cpu"):
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    input_ids = encoded.input_ids.to(device)   
    attention_mask = encoded.attention_mask.to(device)  
    return input_ids, attention_mask

class TextImageDataset(Dataset):
    def __init__(self, images_dir, json_path, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        with open(json_path, "r", encoding="utf-8") as f:
            self.descriptions = json.load(f)

        self.image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.jpeg','.png', '.PNG', '.webp'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        key = os.path.splitext(image_name)[0]
        description = self.descriptions.get(key, "")
        return image, description

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = TextImageDataset("../all_images", "../captions.json", transform=transform)

## for loss
def reconstruct_from_patches(patches, patch_size=16, grid_size=16):
    B, T2, C, p, _ = patches.shape
    assert T2 == grid_size**2, f"Expected T2={grid_size**2}, got {T2}"
    images = []
    for b in range(B):
        rows = []
        for i in range(grid_size):
            row_patches = patches[b, i*grid_size:(i+1)*grid_size]  
            row = torch.cat(list(row_patches), dim=2)
            rows.append(row)
        img = torch.cat(rows, dim=1)
        images.append(img)
    return torch.stack(images, dim=0)


## training
model = Transformer().to(device)
# model.load_state_dict(torch.load("model-1.pth"))

scaler = torch.cuda.amp.GradScaler()
opt = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,           # slightly higher LR for faster convergence
    betas=(0.9, 0.98), # stable for transformers
    eps=1e-8,
    weight_decay=0.01  # helps regularize medium-large model
)
lpips_loss_fn = lpips.LPIPS(net='alex').to(device)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    opt.zero_grad()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,     
    )

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
    
    for step, (images, texts) in enumerate(pbar):
        images = images.to(device)
        text_ids, _ = tokenize_texts(texts, max_len=config.max_text_len, device=device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            image_preds = model(text_ids, images) 
            pred_full = reconstruct_from_patches(image_preds, patch_size=16, grid_size=16)

            loss_l1 = F.l1_loss(pred_full, images)

        loss_lpips = lpips_loss_fn(pred_full.float(), images.float()).mean()

        loss = loss_l1 + 0.15 * loss_lpips
        loss = loss / accum_steps   

        scaler.scale(loss).backward()

        train_loss += (loss.item() * accum_steps)

        if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
            # unscale, clip, step, update scaler
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if (step + 1) % save_each == 0 or (step + 1) == len(dataloader):
            torch.save(model.state_dict(), f"model-{epoch+1}.pth")

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss / len(dataloader):.4f}")

