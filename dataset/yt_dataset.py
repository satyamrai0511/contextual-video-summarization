import os
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class YouTubeCaptionDataset(Dataset):
    def __init__(self, processed_dir="dataset/processed", caption_file="dataset/captions.json", max_len=20):
        self.processed_dir = processed_dir
        self.max_len = max_len

        with open(caption_file, "r") as f:
            self.caption_map = json.load(f)

        self.video_ids = list(self.caption_map.keys())
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid_id = self.video_ids[idx]

        # Load fused features
        fused_path = os.path.join(self.processed_dir, f"{vid_id}.pt")
        fused_seq = torch.load(fused_path)  # [1, 16, 1280]
        fused_seq = fused_seq.squeeze(0)    # [16, 1280]

        # Load and tokenize caption
        caption = self.caption_map[vid_id]
        tokens = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len
        )

        input_ids = tokens["input_ids"].squeeze(0)            # [max_len]
        pad_mask = tokens["attention_mask"].squeeze(0) == 0   # [max_len]

        return {
            "fused_seq": fused_seq,              # [16, 1280]
            "caption_tokens": input_ids,         # [max_len]
            "caption_pad_mask": pad_mask         # [max_len]
        }
