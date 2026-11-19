import clip
import torch
from torchvision import transforms as T

output_dir = "MONET Weights"

def get_monet_preprocess(n_px, mean, std):
    """Exact preprocessing used by MONET/CLIP ViT-L/14"""

    def convert_image_to_rgb(image):
        return image.convert("RGB")

    return T.Compose([
        T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(n_px),
        convert_image_to_rgb,
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def save_monet_checkpoint(model, preprocess_fn, path="monet_vitl14.pth"):
    """Save MONET in standard format compatible with fine-tuning"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'clip_model': 'ViT-L/14',  # Exact name for clip.load()
            'device': 'cuda:0',  # Default device
            'preprocess_params': {
                'n_px': 224,
                'normalize_mean': (0.485, 0.456, 0.406),
                'normalize_std': (0.229, 0.224, 0.225)
            }
        }
    }, path)
    print(f"MONET saved: {path}")
    return path


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load CLIP base model
    model, _ = clip.load("ViT-L/14", device=device, jit=False)

    # Load MONET weights
    print("Loading MONET weights...")
    state_dict = torch.hub.load_state_dict_from_url(
        "https://aimslab.cs.washington.edu/MONET/weight_clip.pt",
        map_location=device
    )
    model.load_state_dict(state_dict)
    model.eval()

    # Create preprocess function (matches fine-tuning usage)
    preprocess_fn = get_monet_preprocess(224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    # Save standard checkpoint
    save_monet_checkpoint(model, preprocess_fn, f"{output_dir}/monet_vitl14.pth")

    print("MONET initialization complete.")
