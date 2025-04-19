import torch
import torchaudio
import cv2
import os

# Config
video_path = 'data/sample_video_1.mp4'
audio_path = 'data/sample_audio_1.wav'
output_dir = 'data/processed'
num_frames = 16

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(output_dir, exist_ok=True)

def extract_video_frames(path, num_frames):
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = torch.linspace(0, total_frames - 1, num_frames).long()
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.tensor(frame_rgb).permute(2, 0, 1) / 255.0
            frames.append(frame_tensor)
    cap.release()
    video_tensor = torch.stack(frames).to(device)
    torch.save(video_tensor, os.path.join(output_dir, 'video_frames.pt'))
    return video_tensor

def extract_audio_features(path):
    waveform, sr = torchaudio.load(path)
    waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.to(device)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(device)
        waveform = resampler(waveform)

    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model = bundle.get_model().to(device)

    with torch.inference_mode():
        features = model(waveform)[0]  # ← FIXED HERE

    torch.save(features, os.path.join(output_dir, 'audio_features.pt'))
    return features

if __name__ == "__main__":
    print("Extracting video frames...")
    v = extract_video_frames(video_path, num_frames)
    print("Video tensor shape:", v.shape)

    print("Extracting audio features...")
    a = extract_audio_features(audio_path)
    print("Audio tensor shape:", a.shape)
