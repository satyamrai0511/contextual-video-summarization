import torch
import torch.nn.functional as F
from torchvision import models, transforms
import os

# --------- Config ---------
video_path = 'data/processed/video_frames.pt'
audio_path = 'data/processed/audio_features.pt'
output_path = 'data/processed/fused_seq.pt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --------- Load Video Frames ---------
video_tensor = torch.load(video_path).to(device)  # [16, 3, H, W]

# Resize video frames to 224x224 for ResNet
video_resized = F.interpolate(video_tensor, size=(224, 224))  # [16, 3, 224, 224]

# Normalize like ImageNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
video_normalized = torch.stack([normalize(f) for f in video_resized])  # [16, 3, 224, 224]

# --------- Get ResNet Embeddings ---------
resnet = models.resnet18(pretrained=True)
resnet.fc = torch.nn.Identity()  # remove classification layer
resnet = resnet.to(device).eval()

with torch.no_grad():
    video_seq = resnet(video_normalized)  # [16, 512]

video_seq = video_seq.unsqueeze(0)  # [1, 16, 512]

# --------- Load Audio Features ---------
audio_tensor = torch.load(audio_path).to(device)  # [1, T_audio, 768]

# Downsample audio to 16 time steps to match video
audio_seq = F.adaptive_avg_pool1d(audio_tensor.permute(0, 2, 1), 16).permute(0, 2, 1)  # [1, 16, 768]

# --------- Fuse Audio + Video ---------
fused_seq = torch.cat([video_seq, audio_seq], dim=-1)  # [1, 16, 1280]

# --------- Save Fused Output ---------
os.makedirs(os.path.dirname(output_path), exist_ok=True)
torch.save(fused_seq, output_path)

print("✅ Fused sequence saved to:", output_path)
print("Fused shape:", fused_seq.shape)
