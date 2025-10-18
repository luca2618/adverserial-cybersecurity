import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import RandomResizedCrop, Normalize, InterpolationMode
import open_clip
import SparseFool
from SparseFool.sparsefool import sparsefool
from tqdm import tqdm
from scipy.ndimage import label

import json
import open_clip
import tqdm
from PIL import Image
import numpy as np


def pixel_edit_cost(original: np.ndarray, modified: np.ndarray) -> int:
    """
    Compute number of changed pixels with reduced cost for continuous regions.
    
    Args:
        original: Original image as uint8 numpy array
        modified: Modified image as uint8 numpy array
        
    Returns:
        Adjusted cost based on number of changed pixels, with reduced cost for continuous regions.
    """
    # Find the difference mask
    diff_mask = np.any(original != modified, axis=-1)
    
    # Label connected components in the difference mask
    labeled_regions, num_features = label(diff_mask)
    
    # Count pixels in each connected region
    total_cost = 0
    for region_id in range(1, num_features + 1):
        region_size = np.sum(labeled_regions == region_id)
        if region_size > 0:
            # Full cost for the first pixel, half cost for the rest
            total_cost += 1 + (region_size - 1) * 0.5
    
    return int(total_cost)


class CLIPSimilarity(nn.Module):
    def __init__(self, model, alpha=6.6231, beta=-1.1130, device='cuda', caption=""):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.caption = caption

        # check device compatibility
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, switching to CPU")
            self.device = 'cpu'
        else:
            self.device = device

        self.rrc = RandomResizedCrop(
            size=(224, 224),
            scale=(0.9, 1.0),
            ratio=(0.75, 1.3333),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True
        )
        self.norm = Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        x = image_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        x = self.rrc(x)
        x = self.norm(x)

        text_tokens = open_clip.tokenize([self.caption]).to(self.device)
        image_features = F.normalize(self.model.encode_image(x), dim=-1)
        text_features = F.normalize(self.model.encode_text(text_tokens), dim=-1)

        similarity = (image_features @ text_features.T).squeeze()
        similarity = torch.sigmoid(similarity * self.alpha + self.beta)

        output = torch.stack([similarity, 1 - similarity]).unsqueeze(0).to(self.device)
        return output




max_iter = 50
lambda_ = 3.

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'  # Use CPU for testing purposes

model, preprocess, tokenizer = open_clip.create_model_and_transforms(
    "ViT-B-32", 
    pretrained="laion2b_s34b_b79k", device=device
)

def load_image_from_pair(pair: dict) -> Image.Image:
    """Load image from the pair dictionary using image_path"""
    return Image.open(pair['image_path']).convert('RGB')


with open('val_pairs.json', 'r') as f:
    val_pairs = json.load(f)

test_losses = []
adversarial_losses = []

for pair in val_pairs[:30]:
    image = load_image_from_pair(pair)
    image_tensor = torch.tensor(np.array(image), dtype=torch.float32) # Change to (C, H, W)

    net = CLIPSimilarity(model, device=device)
    #net.caption = pair['caption']
    net.caption = "cat" + " " + pair['caption']  # Force original class to be in caption
    im = image_tensor.to(device)
    lb = 0
    ub = 255

    x_adv, r, pred_label, fool_label, loops = sparsefool(im, net, lb, ub, lambda_, max_iter, device=device)

    print(f"Original label: {pred_label}, Adversarial label: {fool_label}, Loops: {loops}")

    #pixel cost
    cost = pixel_edit_cost(im.cpu().detach().numpy().astype(np.uint8), x_adv.cpu().detach().numpy().astype(np.uint8))
    print(f"Pixel edit cost: {cost}")

