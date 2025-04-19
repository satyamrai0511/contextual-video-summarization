import torch
from transformers import AutoTokenizer

# ----- Config -----
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FUSED_PATH = "dataset/processed/4fOWQfzWHbc.pt"  # 🔄 Change to any .pt file
MAX_LEN = 20
TOP_K = 10
START_TOKEN = "[CLS]"
END_TOKEN = "[SEP]"

# ----- Load tokenizer -----
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ----- Load full model -----
model = torch.load("checkpoints/full_decoder_model.pth").to(DEVICE)
model.eval()

# ----- Load fused features -----
fused_seq = torch.load(FUSED_PATH).to(DEVICE)  # [1, 16, 1280]

# ----- Start decoding -----
start_id = tokenizer.convert_tokens_to_ids(START_TOKEN)
end_id = tokenizer.convert_tokens_to_ids(END_TOKEN)
generated = [start_id]

for _ in range(MAX_LEN - 1):
    input_ids = torch.tensor([generated], device=DEVICE)
    pad_mask = torch.zeros_like(input_ids).bool()

    with torch.no_grad():
        logits = model(input_ids, fused_seq, pad_mask)

    next_token_logits = logits[0, -1, :]

    # Top-k sampling
    topk = torch.topk(next_token_logits, k=TOP_K)
    probs = torch.nn.functional.softmax(topk.values, dim=-1)
    next_token_id = topk.indices[torch.multinomial(probs, 1).item()].item()

    generated.append(next_token_id)

    # Prevent early stop if not enough tokens generated
    if next_token_id == end_id and len(generated) > 4:
        break

# ----- Debug Output -----
print("🔢 Token IDs:", generated)
print("🧾 Tokens:", tokenizer.convert_ids_to_tokens(generated))

# ----- Final caption -----
caption = tokenizer.decode(generated, skip_special_tokens=True)
print("📝 Generated Caption:\n", caption)
