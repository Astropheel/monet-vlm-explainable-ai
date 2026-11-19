import torch
import clip
from torchvision import transforms as T
from PIL import Image
import os
import matplotlib.pyplot as plt
import time

# Flags
GPUFlag = True
datasetFlag = "dogs"  # "dogs" or "landscape"
finetuningFlag = False
displayFlag = False
saveFlag = True

image_folder = f"POC Data/{datasetFlag}/testing"

if finetuningFlag:
    modelPath = f"MONET Weights/MONET_{datasetFlag}_tuned.pth"
    ID = "finetuned"
else:
    modelPath = "MONET Weights/MONET_vitl14.pth"
    ID = "original"


output_folder = f"MONET Results/{datasetFlag}/{ID}"
os.makedirs(output_folder, exist_ok=True)

# Dataset configuration
if datasetFlag == "landscape":
    text_labels = [
        "trees","clouds","sand","dark","bright",
        "mountain","flower","snow","animals","plants",
        "rocks","grass","water"
    ]
    if finetuningFlag:
        threshold = 5
    else:
        threshold = 9

if datasetFlag == "dogs":
    text_labels = [
        "brown", "black","white","golden",
        "straight fur", "curly fur","long fur","short fur",
        "protruded snout","flat snout","spotted"
    ]
    if finetuningFlag:
        threshold = 5
    else:
        threshold = 9

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


for i, img_name in enumerate(images):
    img_path = os.path.join(image_folder, img_name)

    # INFERENCE
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    text = clip.tokenize(text_labels).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        logits = (image_features @ text_features.T).softmax(dim=-1)

    probs = logits[0].cpu().numpy() * 100  # Convert to % and numpy

    print(f"\n{i + 1}/{len(images)} processed.")


    # Generating Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    img_pil = Image.open(img_path)
    ax1.imshow(img_pil)
    ax1.axis('off')

    ax2.set_xlim(0, 100)
    ax2.set_yticks(range(len(text_labels)))
    ax2.set_yticklabels(text_labels, fontsize=10)
    ax2.set_xlabel('Relevance (%)')
    ax2.set_title('MONET Results')

    # Color bars by threshold
    colors = ['gray' if p < threshold else 'blue' for p in probs]
    ax2.barh(range(len(text_labels)), probs, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    # THRESHOLD LINE
    ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'{threshold}% threshold')
    ax2.legend(loc='upper right')

    # Add percentage text ON EVERY BAR
    for i_label, (label, prob) in enumerate(zip(text_labels, probs)):
        ax2.text(prob + 2, i_label, f'{prob:.1f}%', va='center', fontsize=9, fontweight='bold')

    plt.suptitle(f"MONET Inference on dataset: '{datasetFlag}'\nModel: '{os.path.basename(modelPath)}'", fontsize=16)
    plt.tight_layout()

    if saveFlag:
        plt.savefig(os.path.join(output_folder, f"MONET_{datasetFlag}_{ID}_{i + 1}.jpg"),
                    dpi=150, bbox_inches='tight', format='jpg')
        print(f"Results Figure Saved in '{output_folder}'.")

    if displayFlag:
        plt.show()

print(f"\nFinished Testing. Time elapsed: {time.time() - start_time:.2f} [s].")