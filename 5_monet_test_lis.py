import torch
import clip
from torchvision import transforms as T
from PIL import Image
import os
import matplotlib.pyplot as plt
import time
from support_functions import displayClipResultsV2

# Flags
GPUFlag = True
datasetFlag = "T"  # "T" or "NT"
finetuningFlag = False
displayFigures = False
saveFigures = False

image_folder = f"POC Data/LIS/{datasetFlag}"

if finetuningFlag:
    modelPath = f""
    ID = "finetuned"
else:
    modelPath = "MONET Weights/MONET_vitl14.pth"
    ID = "original"


output_folder = f"MONET Results/{datasetFlag}/{ID}"
os.makedirs(output_folder, exist_ok=True)

# Dataset configuration
if datasetFlag:
    # text_labels = [
    #     "mass with smooth well-defined margins", "mass along tissue planes", "mass with posterior acoustic enhancement",
    #     "mass with ill-defined margins", "mass across tissue planes", "mass with posterior acoustic shadowing", "mass with heterogeneous internal texture"
    # ]
    text_labels = [
        "mass with ill-defined margins", "mass across tissue planes", "mass with posterior acoustic shadowing", "mass with heterogeneous internal texture"
    ]
    # text_labels = [
    #     "mass across tissue planes"
    # ]
    if finetuningFlag:
        threshold = 10
    else:
        threshold = 10


device = "cuda:0" if GPUFlag and torch.cuda.is_available() else "cpu"
print(f"\nMONET-Based Tests of Descriptor / Image Correspondence."
      f"\n\nDataset: '{datasetFlag}'"
      f"\nDescriptors: {text_labels}"
      f"\n\nModel: '{os.path.basename(modelPath)}'"
      f"\nCPU / GPU: '{device}'")


print("\nLoading model...")
start_time = time.time()
saved_model = torch.load(modelPath, map_location=device)
config = saved_model['model_config']
config['device'] = device

model, _ = clip.load(config['clip_model'], device=device, jit=False)
model.load_state_dict(saved_model['model_state_dict'])
model.eval()

def recreate_preprocess(params):
    def convert_image_to_rgb(image):
        return image.convert("RGB")

    return T.Compose([
        T.Resize(params['n_px'], interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(params['n_px']),
        convert_image_to_rgb,
        T.ToTensor(),
        T.Normalize(params['normalize_mean'], params['normalize_std']),
    ])

preprocess = recreate_preprocess(config['preprocess_params'])
print(f"Model loaded and ready. Time elapsed: {time.time() - start_time:.2f} [s].")


images = [f for f in os.listdir(image_folder)
          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"\nFound {len(images)} images. Processing...")

image_hit_count = 0
label_hit_counts = {label: 0 for label in text_labels}
multi_hit_counts = {k: 0 for k in range(1, len(text_labels) + 1)}

for i, img_name in enumerate(images):
    img_path = os.path.join(image_folder, img_name)

    image = Image.open(img_path)
    image = image.crop((0, 120, image.width, image.height))

    imagePreprocessed = preprocess(image).unsqueeze(0).to(device)

    text = clip.tokenize(text_labels).to(device)

    with torch.no_grad():
        image_features = model.encode_image(imagePreprocessed)
        text_features = model.encode_text(text)
        logits = (image_features @ text_features.T).softmax(dim=-1)
    probs = logits[0].cpu().numpy() * 100


    current_image_hits = 0
    for idx, prob in enumerate(probs):
        if prob > threshold:
            label_hit_counts[text_labels[idx]] += 1
            current_image_hits += 1

    for k in range(1, len(text_labels) + 1):
        if current_image_hits >= k:
            multi_hit_counts[k] += 1

    print(f"\n{i + 1}/{len(images)} processed.")

    if saveFigures or displayFigures:
        displayClipResultsV2(image, probs, text_labels, threshold, datasetFlag,
                           output_folder=output_folder, displayFlag=displayFigures, saveFig=saveFigures, img_idx=i)

print(f"\nFinished Testing. Time elapsed: {time.time() - start_time:.2f} [s].")

print("\n--- Final Detection Breakdown ---")
print(f"Dataset ID: '{datasetFlag}' and threshold: '{threshold}%'")

for k, count in multi_hit_counts.items():
    percentage = (count / len(images)) * 100
    print(f"Images with at least {k} detection(s): {count} ({percentage:.2f}%)")

print("\n--- Per-Label Frequency ---")
for label, count in label_hit_counts.items():
    print(f" - {label}: {count} ({count/len(images)*100:.2f}%)")