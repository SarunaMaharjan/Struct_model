import os
import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

class StreamCorpus(IterableDataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        assert os.path.exists(self.file_path), f"Data file not found at {self.file_path}"

    def __iter__(self):
        file_iter = open(self.file_path, 'r', encoding='utf-8')
        for line in file_iter:
            line = line.strip()
            if line:
                ids = self.tokenizer.encode(
                    line,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    truncation=True
                )
                yield torch.tensor(ids).long(), []