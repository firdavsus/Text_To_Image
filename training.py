import torch
from torch import nn
from torch.nn import functional as F
import math
import os
import shutil
from sklearn.model_selection import train_test_split
from VAE import VAE
from ema import EMA
from torch.utils.data import Dataset
import os
import json
from PIL import Image
from torch.utils.data import DataLoader

# def split_dataset(source_dir, train_dir, test_dir, test_size=0.1, random_state=42):
#     image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

#     train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=random_state)

#     os.makedirs(train_dir, exist_ok=True)
#     os.makedirs(test_dir, exist_ok=True)

#     for file in train_files:
#         shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, file))

#     for file in test_files:
#         shutil.copy(os.path.join(source_dir, file), os.path.join(test_dir, file))

#     print(f"Dataset split complete. {len(train_files)} training images, {len(test_files)} test images.")

# source_dir = "images_downloaded"
# train_dir = "data/train/"
# test_dir = "data/test/"

# split_dataset(source_dir, train_dir, test_dir)

##### train 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
# from model import Encoder, Decoder

# Device configuration
device = torch.device('cuda')

# Hyperparameters
num_epochs = 10
learning_rate = 3e-6
beta = 0.00003

batch_size = 8
from torchvision import transforms

to_tensor = transforms.ToTensor()

class TextImageDataset(Dataset):
    def __init__(self, images_dir, json_path, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        with open(json_path, "r", encoding="utf-8") as f:
            self.descriptions = json.load(f)

        self.image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])

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

dataset = TextImageDataset("all_images", "captions.json", transform=transform)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,     
)


model = VAE().to(device)
state = torch.load("checkpoint/old-2.pth", map_location=device)
from collections import OrderedDict

new_state = OrderedDict()
for k, v in state.items():
    new_state[k.replace("module.", "")] = v

model.load_state_dict(new_state, strict=True)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

decay, no_decay = [], []
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if n.endswith(".bias") or "norm" in n.lower() or "groupnorm" in n.lower() or "embedding" in n.lower():
        no_decay.append(p)
    else:
        decay.append(p)

opt = torch.optim.AdamW([
    {"params": decay, "weight_decay": 1e-2},
    {"params": no_decay, "weight_decay": 0.0},
], lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)

mse = nn.MSELoss()
ema = EMA(model, decay=0.9999, device=None)

# Add these hyperparameters
accumulation_steps = 4  # Adjust as needed
effective_batch_size = batch_size * accumulation_steps

train_losses = []
scaler = torch.cuda.amp.GradScaler()

# training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for i, (images, _) in enumerate(dataloader):
        images = images.to(device)

        # Forward pass
        with torch.cuda.amp.autocast(dtype=torch.float16):
            reconstructed, encoded = model(images)
    
            # Compute loss
            recon_loss = nn.MSELoss()(reconstructed, images)
    
            # Extract mean and log_variance from encoded
            mean, log_variance = torch.chunk(encoded.float(), 2, dim=1)
            kl_div = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())
            loss = recon_loss + beta * kl_div
    
            # Normalize the loss to account for accumulation
            loss = loss / accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            scaler.step(opt) 
            scaler.update() 
            opt.zero_grad(set_to_none=True) 
            ema.update(model)

        train_loss += loss.item() * accumulation_steps

        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], '
              f'Loss: {loss.item()*accumulation_steps:.4f}, Recon Loss: {recon_loss.item():.4f}, KL Div: {kl_div.item():.4f}')


        if i % 150 ==0:
            with torch.no_grad():
                # Take the first image from the batch
                sample_image = images[0].unsqueeze(0)
                sample_reconstructed = model(sample_image)[0]
    
                sample_image = (sample_image * 0.5) + 0.5
                sample_reconstructed = (sample_reconstructed * 0.5) + 0.5
    
                torchvision.utils.save_image(sample_reconstructed, 'reconstructed.png')

    train_losses.append(train_loss / len(dataloader))
  # Save the model checkpoint
    torch.save(model.state_dict(), f'checkpoint/vae_model_epoch_{epoch+1}.pth')

print('Training finished!')