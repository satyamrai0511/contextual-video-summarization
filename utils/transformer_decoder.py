import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x


class TransformerDecoderModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.1, max_len=100):
        super().__init__()

        self.encoder_proj = nn.Linear(1280, d_model)  # Project fused_seq [B, 16, 1280] → [B, 16, 512]

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # For shape: [B, T, D]
        )

        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz, device=self.fc_out.weight.device) * float('-inf'), diagonal=1)

    def forward(self, tgt, memory, tgt_pad_mask=None):
        """
        tgt: [B, T_dec]         - token ids
        memory: [B, T_enc, 1280] - fused_seq
        tgt_pad_mask: [B, T_dec] - padding mask
        """
        # Embed target tokens
        tgt_emb = self.token_embedding(tgt)  # [B, T_dec, D]
        tgt_emb = self.pos_encoder(tgt_emb)

        # Project encoder memory to match decoder's expected input dim
        memory_proj = self.encoder_proj(memory)  # [B, T_enc, D]

        # Create masked attention for autoregressive decoding
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)

        # Transformer decoding
        out = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory_proj,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )

        # Output logits
        logits = self.fc_out(out)  # [B, T_dec, vocab_size]
        return logits
