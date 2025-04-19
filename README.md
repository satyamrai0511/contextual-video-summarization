# 🎬 Contextual Video Summarization with Multimodal Transformers

This project generates context-aware captions for videos by fusing visual and audio features and decoding them using a transformer-based language model. It supports feature extraction, model training, inference using top-k sampling, and runs fully end-to-end from raw YouTube video to text output.

## 📁 Folder Structure

```
contextual_video_summarization/
├── .venv/                       # Virtual environment (not pushed to GitHub)
├── checkpoints/                 # Trained model weights (.pth)
│   ├── decoder_latest.pth
│   └── full_decoder_model.pth
├── dataset/
│   ├── raw/                     # Downloaded .mp4 videos
│   ├── processed/               # Extracted fused features (.pt)
│   ├── captions.json            # Video ID → caption mapping
│   └── yt_dataset.py            # Custom dataset class
├── scripts/
│   ├── download_video.py        # Downloads YouTube videos
│   └── extract_dataset_features.py # Extracts audio + visual features
├── utils/
│   ├── extract_features.py
│   ├── fuse_modalities.py
│   └── transformer_decoder.py
├── inference_generate.py        # Caption generation with top-k sampling
├── train_model.py               # Final training script
├── train_dummy_decoder.py       # Optional: early test version
├── test_dataset_loader.py       # Dataset loader verification
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## 🔧 Installation

1. Clone the repository:

```bash
git clone https://github.com/satyamrai0511/contextual_video_summarization
cd contextual_video_summarization
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/Scripts/activate  # On Windows
```

3. Install all dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install opencv-python
pip install yt-dlp
pip install soundfile
```

## 🗂️ Dataset Setup

### 1. Add YouTube Video Links

Edit `scripts/download_video.py`:

```python
videos = {
  "VIDEO_ID_1": "https://youtu.be/...",
  "VIDEO_ID_2": "https://youtu.be/...",
  ...
}
```

### 2. Add Captions

Update `dataset/captions.json` to match the video IDs:

```json
{
  "VIDEO_ID_1": "Short summary of the video content.",
  "VIDEO_ID_2": "Another caption here.",
  ...
}
```

## 🛠️ Run Preprocessing

### 1. Download Videos

```bash
python scripts/download_video.py
```

### 2. Extract Fused Features

```bash
python scripts/extract_dataset_features.py
```

This saves `.pt` tensors in `dataset/processed/` combining audio + visual embeddings.

## 🧠 Train the Model

```bash
python train_model.py
```

- Uses `transformer_decoder.py` from `utils/`
- Trains on fused features and matched captions
- Saves weights to `checkpoints/full_decoder_model.pth`

## 🧪 Run Inference

Update `FUSED_PATH` inside `inference_generate.py` to the desired `.pt` file:

```python
FUSED_PATH = "dataset/processed/VIDEO_ID.pt"
```

Then run:

```bash
python inference_generate.py
```

Sample Output:

```
📝 Generated Caption:
a on on how.
```

> Top-k sampling used to improve diversity and avoid early stopping.

## 🔬 Evaluation

- Generation quality is limited by the number of training examples (e.g., 10–20)
- Captions improve with more data and training time
- Repetition and short outputs are expected in low-resource scenarios

## 📦 Requirements

- Python 3.9+
- torch
- torchaudio
- torchvision
- transformers
- opencv-python
- yt-dlp
- soundfile

## 🚀 Future Improvements

- Beam search decoding
- BLEU/CIDEr scoring
- Whisper integration for ASR
- Streamlit/Gradio demo app
- More robust caption alignment pipeline

## 📜 License

MIT License. See `LICENSE` file.

## 🤝 Acknowledgments

- TEDx Talks YouTube Channel (for video content)
- HuggingFace Transformers
- Wav2Vec2 Pretrained Audio Models
- OpenCV + torchvision
