import torch
from torchvision.transforms.functional import to_pil_image
from transformers import GPT2Tokenizer
from model import Transformer, config

# tokenizer + model setup (as you had)
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
    return encoded.input_ids.to(device), encoded.attention_mask.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer().to(device)
model.load_state_dict(torch.load("model-1.pth", map_location=device))
model.eval()

def generate_image_autoregressive(prompt, model, device=device):
    """
    Generates a 256x256 image from a prompt using the model.
    The model now returns the full image [B, 3, H, W] directly.
    """
    model.eval()
    
    # tokenize text
    text_ids, _ = tokenize_texts([prompt], max_len=config.max_text_len, device=device)
    
    with torch.no_grad():
        # pass dummy full_image as input if your model requires it, else just text_ids
        # assuming model returns [B, 3, H, W]
        preds = model(text_ids)  # [B, 3, H, W] expected

        if preds.dim() != 4 or preds.shape[1] != 3:
            raise RuntimeError(f"Unexpected model output shape {preds.shape}, expected [B,3,H,W]")

        # take first image in batch
        img_tensor = preds[0].cpu()

        # denormalize assuming [-1,1] -> [0,1]
        img_tensor = (img_tensor + 1.0) / 2.0
        img_tensor = img_tensor.clamp(0, 1)

        # convert to PIL
        pil_img = to_pil_image(img_tensor)
        return pil_img

if __name__ == "__main__":
    while True:
        prompt = input("Enter a prompt: ")
        img = generate_image_autoregressive(prompt, model)
        img.show()