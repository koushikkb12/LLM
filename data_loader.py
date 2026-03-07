from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import IterableDataset

class StreamingDataset(IterableDataset):
    def __init__(self, dataset_name, tokenizer_name, split="train", buffer_size=10000, block_size=1024, tokenizer_use_fast=False):
        self.dataset = load_dataset(dataset_name, split=split, streaming=True)
        self.shuffled = self.dataset.shuffle(buffer_size=buffer_size)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=tokenizer_use_fast)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.block_size = block_size

    def __iter__(self):
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        for sample in self.shuffled:
            tokenized = self.tokenizer(
                sample['text'],
                truncation=True,
                max_length=self.block_size,
                padding="max_length",
                return_tensors=None,
            )
            input_ids = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
            # For causal LM: labels = input_ids, with -100 where we should not compute loss (padding)
            labels = [tid if attention_mask[i] else -100 for i, tid in enumerate(input_ids)]
            yield {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            }

def get_dataloader(dataset_name, tokenizer_name, batch_size, block_size, buffer_size=10000, tokenizer_use_fast=False):
    dataset = StreamingDataset(dataset_name, tokenizer_name, buffer_size=buffer_size, block_size=block_size, tokenizer_use_fast=tokenizer_use_fast)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)