import os
from PIL import Image
import matplotlib.pyplot as plt

def displayClipResults(image_path, probs, text_labels, threshold, datasetFlag, output_folder=None,
                       displayFlag=False,saveFig=False, img_idx=0):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    img_pil = Image.open(image_path)
    ax1.imshow(img_pil)
    ax1.axis('off')

    ax2.set_xlim(0, 100)
    ax2.set_yticks(range(len(text_labels)))
    ax2.set_yticklabels(text_labels, fontsize=10)
    ax2.set_xlabel('Relevance (%)')
    ax2.set_title('CLIP Results')

    # Color bars by threshold
    colors = ['gray' if p < threshold else 'blue' for p in probs]
    ax2.barh(range(len(text_labels)), probs, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    # THRESHOLD LINE
    ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'{threshold}% threshold')
    ax2.legend(loc='upper right')

    # Add percentage text ON EVERY BAR
    for i_label, (label, prob) in enumerate(zip(text_labels, probs)):
        ax2.text(prob + 2, i_label, f'{prob:.1f}%', va='center', fontsize=9, fontweight='bold')

    plt.suptitle(f"CLIP Inference on dataset: '{datasetFlag}'\nModel: 'ViT-L/14'", fontsize=16)
    plt.tight_layout()

    # SAVE OPTION
    if saveFig and output_folder:
        os.makedirs(output_folder, exist_ok=True)
        save_path = os.path.join(output_folder, f"CLIP_{datasetFlag}_{img_idx + 1}.jpg")
        plt.savefig(save_path, dpi=150, bbox_inches='tight', format='jpg')
        print(f"Results Figure Saved: '{save_path}'")

    if displayFlag:
        plt.show()

def displayClipResultsV2(image, probs, text_labels, threshold, datasetFlag, output_folder=None,
                       displayFlag=False,saveFig=False, img_idx=0):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.imshow(image)
    ax1.axis('off')

    ax2.set_xlim(0, 100)
    ax2.set_yticks(range(len(text_labels)))
    ax2.set_yticklabels(text_labels, fontsize=10)
    ax2.set_xlabel('Relevance (%)')
    ax2.set_title('CLIP Results')

    # Color bars by threshold
    colors = ['gray' if p < threshold else 'blue' for p in probs]
    ax2.barh(range(len(text_labels)), probs, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    # THRESHOLD LINE
    ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'{threshold}% threshold')
    ax2.legend(loc='upper right')

    # Add percentage text ON EVERY BAR
    for i_label, (label, prob) in enumerate(zip(text_labels, probs)):
        ax2.text(prob + 2, i_label, f'{prob:.1f}%', va='center', fontsize=9, fontweight='bold')

    plt.suptitle(f"CLIP Inference on dataset: '{datasetFlag}'\nModel: 'ViT-L/14'", fontsize=16)
    plt.tight_layout()

    # SAVE OPTION
    if saveFig and output_folder:
        os.makedirs(output_folder, exist_ok=True)
        save_path = os.path.join(output_folder, f"CLIP_{datasetFlag}_{img_idx + 1}.jpg")
        plt.savefig(save_path, dpi=150, bbox_inches='tight', format='jpg')
        print(f"Results Figure Saved: '{save_path}'")

    if displayFlag:
        plt.show()