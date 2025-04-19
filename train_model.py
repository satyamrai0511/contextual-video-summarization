import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset.yt_dataset import YouTubeCaptionDataset
from utils.transformer_decoder import TransformerDecoderModel

# ----- Config -----
BATCH_SIZE = 2
MAX_LEN = 20
EPOCHS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----- Load Dataset -----
dataset = YouTubeCaptionDataset(max_len=MAX_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----- Model -----
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = TransformerDecoderModel(
    vocab_size=tokenizer.vocab_size,
    d_model=512,
    nhead=8,
    num_layers=4,
    max_len=MAX_LEN
).to(DEVICE)

# ----- Optimizer and Loss -----
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# ----- Training Loop -----
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in loader:
        fused_seq = batch["fused_seq"].to(DEVICE)                          # [B, 16, 1280]
        tgt = batch["caption_tokens"].to(DEVICE)                          # [B, max_len]
        pad_mask = batch["caption_pad_mask"].to(DEVICE)                   # [B, max_len]

        # Shift targets for decoder input/output
        tgt_in = tgt[:, :-1]      # input to decoder
        tgt_out = tgt[:, 1:]      # expected output
        pad_mask_in = pad_mask[:, :-1]

        # Forward pass
        logits = model(tgt_in, fused_seq, pad_mask_in)                    # [B, T, vocab_size]
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "checkpoints/decoder_latest.pth")
torch.save(model, "checkpoints/full_decoder_model.pth")
