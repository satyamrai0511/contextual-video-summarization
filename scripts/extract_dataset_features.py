import os
import torch
import torch.nn.functional as F
import torchaudio
import cv2
import subprocess
import json
from torchvision import models, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------- Config --------
video_dir = "dataset/raw"
output_dir = "dataset/processed"
os.makedirs(output_dir, exist_ok=True)

# -------- Load captions --------
with open("dataset/captions.json", "r") as f:
    captions = json.load(f)

# -------- Models --------
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device).eval()

wav2vec = torchaudio.pipelines.WAV2VEC2_BASE.get_model().to(device).eval()

# -------- Preprocessing --------
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def extract_video_frames(path, num_frames=16):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = torch.linspace(0, total - 1, num_frames).long()
    frames = []
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.tensor(frame_rgb).permute(2, 0, 1) / 255.0
            frames.append(frame_tensor)
    cap.release()

    video = torch.stack(frames).to(device)  # [16, 3, H, W]
    video = F.interpolate(video, size=(224, 224))  # Resize
    video = torch.stack([normalize(f) for f in video])  # Normalize

    with torch.no_grad():
        vid_emb = resnet(video)  # [16, 512]
    return vid_emb.unsqueeze(0)  # [1, 16, 512]

def extract_audio_embeddings(video_path):
    audio_path = "temp_audio.wav"
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", "16000",
        "-vn", audio_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.to(device)

    with torch.no_grad():
        audio = wav2vec(waveform)[0]  # [1, T, 768]
        pooled = F.adaptive_avg_pool1d(audio.permute(0, 2, 1), 16).permute(0, 2, 1)  # [1, 16, 768]

    os.remove(audio_path)
    return pooled

# -------- Process All Videos --------
for vid_id in captions.keys():
    video_path = os.path.join(video_dir, f"{vid_id}.mp4")
    if not os.path.exists(video_path):
        print(f"Missing video: {vid_id}.mp4")
        continue

    print(f"Processing {vid_id}...")
    try:
        video_feat = extract_video_frames(video_path)       # [1, 16, 512]
        audio_feat = extract_audio_embeddings(video_path)   # [1, 16, 768]
        fused = torch.cat([video_feat, audio_feat], dim=-1) # [1, 16, 1280]
        torch.save(fused, os.path.join(output_dir, f"{vid_id}.pt"))
        print(f"✅ Saved: {vid_id}.pt → shape: {fused.shape}")
    except Exception as e:
        print(f"❌ Failed processing {vid_id}: {str(e)}")
