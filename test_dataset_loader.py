from dataset.yt_dataset import YouTubeCaptionDataset
from torch.utils.data import DataLoader

# Create dataset instance
dataset = YouTubeCaptionDataset()

# Create DataLoader (batch size 2 since we have 2 samples)
loader = DataLoader(dataset, batch_size=2)

# Grab one batch and print shapes
for batch in loader:
    print("Fused Sequence:", batch["fused_seq"].shape)            # [B, 16, 1280]
    print("Caption Tokens:", batch["caption_tokens"].shape)      # [B, max_len]
    print("Padding Mask:", batch["caption_pad_mask"].shape)      # [B, max_len]
    break
