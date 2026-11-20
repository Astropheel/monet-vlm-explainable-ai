import torch
import torch.nn as nn
import json
from torch.utils.data import Dataset, DataLoader
import clip
from PIL import Image
from torchvision import transforms as T
import os
import time
import re

GPUFlag = True

datasetFlag = "dogs" # landscape, dogs...

trainingImageFolder = f"data/{datasetFlag}/training"
jsonDescriptors = f"data/{datasetFlag}/training/labelsTexts.json"

modelPath = "MONET Weights/MONET_vitl14.pth"
output_dir = "MONET Weights"
outputModelName = f"MONET_{datasetFlag}_tuned.pth"

batch_size = 8
epochs = 10

device = "cuda:0" if GPUFlag and torch.cuda.is_available() else "cpu"
os.makedirs(output_dir, exist_ok=True)
print(f"\nMONET Fine-Tuning."
      f"\n\nDataset: '{datasetFlag}'"
      f"\nModel: '{os.path.basename(modelPath)}'"
      f"\nCPU / GPU: '{device}'")


def get_monet_preprocess(n_px, mean, std):
    def convert_image_to_rgb(image):
        return image.convert("RGB")

    return T.Compose([
        T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(n_px),
        convert_image_to_rgb,
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


# DATA LOADING (Unchanged)
print("\nLoading dataset...")
with open(jsonDescriptors, 'r') as f:
    all_tags = json.load(f)

# Create text prompts
text_prompts = []
for tags_list in all_tags:
    prompt = f"{datasetFlag} photo containing {', '.join(tags_list)}"
    text_prompts.append(prompt)

image_files = sorted(
    [f for f in os.listdir(trainingImageFolder) if f.lower().endswith('.jpg')],
    key=lambda f: int(re.search(r'\((\d+)\)', f).group(1))
)

for f, p in list(zip(image_files, text_prompts))[:5]:
    print(f, "->", p)
    

dataset_pairs = []
for img_file, prompt in zip(image_files, text_prompts):
    dataset_pairs.append((os.path.join(trainingImageFolder, img_file), prompt))

print(f"Loaded {len(dataset_pairs)} image-text pairs.")
print(f"Sample prompt generated from training descriptors:\n'{dataset_pairs[0][1]}'")


class TextDataset(Dataset):
    def __init__(self, pairs, preprocess):
        self.pairs = pairs
        self.preprocess = preprocess

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, prompt = self.pairs[idx]
        image = self.preprocess(Image.open(img_path).convert("RGB"))
        text = clip.tokenize(prompt, truncate=True)[0]  # Tokenize text
        return image, text


# LOAD MONET
print("\nLoading MONET...")
checkpoint = torch.load(modelPath, map_location=device)
config = checkpoint['model_config']

preprocess_fn = get_monet_preprocess(
    config['preprocess_params']['n_px'],
    config['preprocess_params']['normalize_mean'],
    config['preprocess_params']['normalize_std']
)

model, _ = clip.load(config['clip_model'], device=device, jit=False)
model.load_state_dict(checkpoint['model_state_dict'])

model.float()

# FREEZE IMAGE, TUNE TEXT
for name, param in model.named_parameters():
    if "visual" in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

print("Image frozen | Text trainable | Float32 active")

# SAFE OPTIMIZER
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.01)

# CONTRASTIVE LOSS
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

train_loader = DataLoader(TextDataset(dataset_pairs, preprocess_fn),
                          batch_size=batch_size, shuffle=True)

# TRAINING
startTime_Training = time.time()
print(f"\n=== FINE-TUNING. Max Epochs = {epochs}. Early Stop = {False} ===")
for epoch in range(epochs):
    startTime_Epoch = time.time()
    model.train()
    total_loss = 0.0

    for images, texts in train_loader:
        images, texts = images.to(device), texts.to(device)
        optimizer.zero_grad()

        # Forward pass (image + text)
        logits_per_image, logits_per_text = model(images, texts)

        # Symmetric loss
        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
        loss = (loss_img(logits_per_image, ground_truth) +
                loss_txt(logits_per_text, ground_truth)) / 2

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1:2d} | Loss: {avg_loss:.4f} | Time/Epoch: {time.time() - startTime_Epoch:.2f}[s] | Total Time Elapsed: {time.time() - startTime_Training:.2f}[s]")

# SAVE TUNED MODEL
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': config,
    'preprocess_params': config['preprocess_params']
}, os.path.join(output_dir, outputModelName))
print(f"\nSaved tuned model in '{output_dir}/MONET_{datasetFlag}_tuned.pth'")
