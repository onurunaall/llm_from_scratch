import os
import torch
import torch.nn as nn
from data import prepare_tokenizers, get_dataloaders
from model import Transformer
from utils import create_padding_mask, create_look_ahead_mask
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    os.makedirs("./tokenizer_en", exist_ok=True)
    os.makedirs("./tokenizer_es", exist_ok=True)
    
    # Load a subset to build tokenizers
    train_dataset_full = load_dataset("Helsinki-NLP/opus-100", "en-es", split='train')
    raw_train, _ = torch.utils.data.random_split(train_dataset_full, [1500, len(train_dataset_full)-1500])
    tokenizer_en, tokenizer_es = prepare_tokenizers(raw_train)
    
    train_loader, val_loader = get_dataloaders(tokenizer_en, tokenizer_my)
    source_vocab_size = tokenizer_en.get_vocab_size()
    target_vocab_size = tokenizer_es.get_vocab_size()
    pad_id = tokenizer_es.token_to_id("[PAD]")

    # Model parameters
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    dropout_rate = 0.1

    model = Transformer(source_vocab_size, target_vocab_size, d_model, num_heads, d_ff, num_layers, dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            encoder_input = batch["encoder_input"].to(device)  # (batch, seq_len)
            decoder_input = batch["decoder_input"].to(device)
            target_label = batch["target_label"].to(device)

            enc_mask = create_padding_mask(encoder_input, pad_id)  # (batch, 1, 1, seq_len)
            dec_pad_mask = create_padding_mask(decoder_input, pad_id)  # (batch, 1, 1, seq_len)
            look_ahead_mask = create_look_ahead_mask(decoder_input.size(1), device)  # (seq_len, seq_len)
            # Expand look-ahead mask to (batch, 1, seq_len, seq_len)
            look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(1)
            dec_mask = dec_pad_mask & look_ahead_mask

            optimizer.zero_grad()
            output = model(encoder_input, decoder_input, enc_mask, dec_mask)  # (batch, seq_len, target_vocab_size)
            loss = loss_fn(output.view(-1, target_vocab_size), target_label.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    main()