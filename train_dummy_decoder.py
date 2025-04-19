import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from utils.transformer_decoder import TransformerDecoderModel

# -------- Config --------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
fused_path = 'data/processed/fused_seq.pt'
caption = "A man is giving a talk on stage."
max_len = 20

# -------- Load fused features --------
fused_seq = torch.load(fused_path).to(device)  # [1, 16, 1280]

# -------- Tokenize caption --------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer(caption, return_tensors="pt", padding='max_length', truncation=True, max_length=max_len)

tgt_tokens = tokens['input_ids'].to(device)          # [1, max_len]
tgt_pad_mask = tokens['attention_mask'] == 0         # [1, max_len]
tgt_pad_mask = tgt_pad_mask.to(device)

# -------- Shift tokens for decoder input/output --------
tgt_in = tgt_tokens[:, :-1]  # input to decoder (exclude last)
tgt_out = tgt_tokens[:, 1:]  # target output (exclude first)

# -------- Model --------
model = TransformerDecoderModel(
    vocab_size=tokenizer.vocab_size,
    d_model=512,
    nhead=8,
    num_layers=4,
    max_len=max_len
).to(device)

# -------- Loss and Optimizer --------
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -------- Forward pass --------
logits = model(tgt_in, fused_seq, tgt_pad_mask[:, :-1])  # [1, T, vocab_size]

# -------- Compute loss --------
loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.view(-1))
print("Dummy loss:", loss.item())

# -------- Backward pass --------
loss.backward()
optimizer.step()
