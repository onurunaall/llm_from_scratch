import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_ds_iterator(raw_dataset, lang):
    for data in raw_dataset:
        yield data['translation'][lang]

def prepare_tokenizers(raw_train_dataset):
    # Directories
    os.makedirs("./tokenizer_en", exist_ok=True)
    os.makedirs("./tokenizer_es", exist_ok=True)
    tokenizer_en_path = "./tokenizer_en/tokenizer_en.json"
    tokenizer_my_path = "./tokenizer_es/tokenizer_es.json"
    
    # English tokenizer
    if not os.path.exists(tokenizer_en_path):
        tokenizer_en = Tokenizer(BPE(unk_token="[UNK]"))
        trainer_en = BpeTrainer(min_frequency=2, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
        tokenizer_en.pre_tokenizer = Whitespace()
        tokenizer_en.train_from_iterator(get_ds_iterator(raw_train_dataset, "en"), trainer=trainer_en)
        tokenizer_en.save(tokenizer_en_path)
    else:
        tokenizer_en = Tokenizer.from_file(tokenizer_en_path)
    
    # Spanish tokenizer
    if not os.path.exists(tokenizer_es_path):
        tokenizer_es = Tokenizer(BPE(unk_token="[UNK]"))
        trainer_es = BpeTrainer(min_frequency=2, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
        tokenizer_es.pre_tokenizer = Whitespace()
        tokenizer_es.train_from_iterator(get_ds_iterator(raw_train_dataset, "ms"), trainer=trainer_es)
        tokenizer_es.save(tokenizer_es_path)
    else:
        tokenizer_es = Tokenizer.from_file(tokenizer_es_path)
    
    return tokenizer_en, tokenizer_es

class EncodeDataset(Dataset):
    def __init__(self, raw_dataset, tokenizer_en, tokenizer_my, max_seq_len):
        super().__init__()
        self.raw_dataset = raw_dataset
        self.tokenizer_en = tokenizer_en
        self.tokenizer_my = tokenizer_my
        self.max_seq_len = max_seq_len
        # Get special token IDs (assumes both use the same specials)
        self.cls_id = self.tokenizer_my.token_to_id("[CLS]")
        self.sep_id = self.tokenizer_my.token_to_id("[SEP]")
        self.pad_id = self.tokenizer_my.token_to_id("[PAD]")

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        raw_text = self.raw_dataset[index]
        source_text = raw_text['translation']['en']
        target_text = raw_text['translation']['es']

        source_ids = self.tokenizer_en.encode(source_text).ids
        target_ids = self.tokenizer_my.encode(target_text).ids

        # Build sequences with special tokens
        encoder_ids = [self.cls_id] + source_ids + [self.sep_id]
        decoder_ids = [self.cls_id] + target_ids
        target_label_ids = target_ids + [self.sep_id]

        # Pad sequences to max_seq_len
        encoder_ids = self.pad_sequence(encoder_ids, self.max_seq_len, self.pad_id)
        decoder_ids = self.pad_sequence(decoder_ids, self.max_seq_len, self.pad_id)
        target_label_ids = self.pad_sequence(target_label_ids, self.max_seq_len, self.pad_id)

        return {
            'encoder_input': torch.tensor(encoder_ids, dtype=torch.long),
            'decoder_input': torch.tensor(decoder_ids, dtype=torch.long),
            'target_label': torch.tensor(target_label_ids, dtype=torch.long),
            'source_text': source_text,
            'target_text': target_text
        }

    def pad_sequence(self, seq, max_len, pad_id):
        if len(seq) < max_len:
            seq.extend([pad_id] * (max_len - len(seq)))
        else:
            seq = seq[:max_len]
        return seq

def get_dataloaders(tokenizer_en, tokenizer_my, max_seq_len=155, batch_size_train=5, batch_size_val=1):
    # Load full dataset from Hugging Face
    train_dataset = load_dataset("Helsinki-NLP/opus-100", "en-es", split='train')
    validation_dataset = load_dataset("Helsinki-NLP/opus-100", "en-es", split='validation')

    # Limit sizes for faster training
    raw_train, _ = random_split(train_dataset, [1500, len(train_dataset) - 1500])
    raw_val, _ = random_split(validation_dataset, [50, len(validation_dataset) - 50])

    train_ds = EncodeDataset(raw_train, tokenizer_en, tokenizer_my, max_seq_len)
    val_ds = EncodeDataset(raw_val, tokenizer_en, tokenizer_my, max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size_val, shuffle=True)
    return train_loader, val_loader