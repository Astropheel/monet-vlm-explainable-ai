import torch
import clip
from PIL import Image
import os
from support_functions import *

datasetID = "dogs"  # "dogs" or "landscape"
image_folder = f"POC Data/{datasetID}/testing"
output_folder=f"CLIP Results/{datasetID}"
displayFigures = True
saveFigures = False



if datasetID == "landscape":
    text_labels = ["trees", "clouds", "sand", "dark", "bright", "mountain", "flower", "snow", "animals", "plants",
                   "rocks", "grass", "water"]
    threshold = 5
elif datasetID == "dogs":
    text_labels = ["brown", "black", "white", "golden", "straight fur", "curly fur", "long fur", "short fur",
                   "protruded snout", "flat snout", "spotted"]
    threshold = 9

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model, preprocess = clip.load("ViT-L/14", device=device)

images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
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


    displayClipResults(img_path, probs, text_labels, threshold, datasetID,
                       output_folder=output_folder, displayFlag=displayFigures, saveFig=saveFigures, img_idx=i)

print(f"\n Results saved to: {output_folder}")
