import argparse
import math
import random
import time
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoTokenizer

from data_hindi import StreamCorpus
from struct_xlmr import StructXLMRoberta

# ---------------- Checkpoint Utilities ----------------
def model_save(fn, model, optimizer, scheduler, global_step, stored_loss):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'global_step': global_step,
        'stored_loss': stored_loss
    }
    torch.save(state, fn)

def model_load(fn, model, device):
    checkpoint = torch.load(fn, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(">>> Model weights loaded. Optimizer will be re-initialized. <<<")
    global_step = checkpoint.get('global_step', 0)
    stored_loss = checkpoint.get('stored_loss', float('inf'))
    print(f"Resuming from step {global_step} with best validation loss {stored_loss:.2f}")
    return global_step, stored_loss

# ---------------- Data Utilities ----------------
def collate_batch(batch, pad_token_id):
    sents, heads = zip(*batch)
    padded_sents = torch.nn.utils.rnn.pad_sequence(
        sents, batch_first=True, padding_value=pad_token_id
    )
    return padded_sents, heads

def mask_data(data, pad_token, unk_token, mask_token, mask_bernoulli, device):
    input_data = data.clone()
    mask = mask_bernoulli.sample(input_data.shape).to(device).bool()
    mask &= (input_data != pad_token) & (input_data != unk_token)
    if mask.sum() == 0:
        valid_positions = (input_data != pad_token) & (input_data != unk_token)
        if valid_positions.sum() > 0:
            indices = valid_positions.nonzero(as_tuple=False)
            rand_idx = torch.randint(len(indices), (1,))
            coord_to_mask = indices[rand_idx]
            mask[coord_to_mask[0, 0], coord_to_mask[0, 1]] = True
    targets = input_data.clone()
    targets[~mask] = -100
    input_data[mask] = mask_token
    return input_data, targets

@torch.no_grad()
def evaluate(model, loader, device):
    tokenizer = loader.dataset.tokenizer
    pad_token, unk_token, mask_token = tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.mask_token_id
    mask_bernoulli = torch.distributions.Bernoulli(0.15)
    model.eval()
    total_loss = 0
    total_count = 0
    max_eval_batches = 200
    for i, (data, heads) in enumerate(loader):
        if i >= max_eval_batches:
            break
        data = data.to(device)
        input_data, targets = mask_data(data, pad_token, unk_token, mask_token, mask_bernoulli, device)
        loss, _ = model(input_data, targets)
        if not torch.isnan(loss):
            count = (targets != -100).float().sum().item()
            total_loss += loss.item() * count
            total_count += count
    model.train()
    return total_loss / total_count if total_count > 0 else float('inf')

# ---------------- Training Loop ----------------
def train(args, model, optimizer, scheduler, train_loader, valid_loader, device):
    global_step = 0
    stored_loss = float('inf')
    if args.resume:
        global_step, stored_loss = model_load(args.resume, model, device)

    tokenizer = train_loader.dataset.tokenizer
    pad_token, unk_token, mask_token = tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.mask_token_id
    mask_bernoulli = torch.distributions.Bernoulli(args.mask_rate)
    model.train()
    total_loss = 0
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"--- Starting Epoch {epoch}/{args.epochs} ---")
        for data, heads in train_loader:
            data = data.to(device)
            input_data, targets = mask_data(data, pad_token, unk_token, mask_token, mask_bernoulli, device)
            optimizer.zero_grad()
            loss, _ = model(input_data, targets)
            if torch.isnan(loss):
                print(f"Warning: NaN loss at step {global_step}. Skipping update.")
                continue
            
            loss.backward()

            with torch.no_grad():
                parser_grad_norm = sum(p.grad.norm().item() for p in model.parser.parameters() if p.grad is not None)
                # --- THIS IS THE CORRECTED LINE ---
                gate_grad_norm = sum(p.grad.norm().item() for n, p in model.rear_layers.named_parameters() if "struct_gate" in n and p.grad is not None)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            global_step += 1
            if global_step > 0 and global_step % args.log_interval == 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                with torch.no_grad():
                    parser_weight_norm = sum(p.norm().item() for p in model.parser.parameters())
                    gate_val_sample = model.rear_layers[0].attention.struct_gate.data.mean().item()

                print(f"| Step {global_step} | loss {cur_loss:.2f} | ppl {math.exp(cur_loss):.2f} "
                      f"| parser_w_norm {parser_weight_norm:.2f} | parser_g_norm {parser_grad_norm:.2e} "
                      f"| gate_val {gate_val_sample:.2f} | gate_g_norm {gate_grad_norm:.2e}")
                total_loss = 0
                start_time = time.time()
            if global_step > 0 and global_step % args.validation_interval == 0:
                print("\n--- Running Periodic Validation ---")
                val_loss = evaluate(model, valid_loader, device)
                print(f"| Step {global_step} | valid loss {val_loss:.2f} | valid ppl {math.exp(val_loss):.2f}")
                print("--- Saving Checkpoint ---")
                model_save(args.checkpoint_path, model, optimizer, scheduler, global_step, stored_loss)
                if val_loss < stored_loss:
                    print(">>> New best validation loss. Saving best model. <<<")
                    stored_loss = val_loss
                    model_save(args.save, model, optimizer, scheduler, global_step, stored_loss)
                print("-" * 50)
    return stored_loss

# ---------------- Main Script ----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Masked LM training with Struct-XLM-R')
    parser.add_argument('--base_lr', type=float, default=1e-5)
    parser.add_argument('--parser_lr', type=float, default=1e-6)
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--valid_file', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, default='hindi_RobertaTok')
    parser.add_argument('--model_path', type=str, default='xlm-roberta-base-local')
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--mask_rate', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--log_interval', type=int, default=200)
    parser.add_argument('--save', type=str, default='struct_xlmr_final.pt')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pt')
    parser.add_argument('--validation_interval', type=int, default=2500)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of CPU workers for data loading.')
    args = parser.parse_args()
    
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    train_corpus = StreamCorpus(file_path=args.train_file, tokenizer=tokenizer)
    valid_corpus = StreamCorpus(file_path=args.valid_file, tokenizer=tokenizer)
    train_loader = DataLoader(train_corpus, batch_size=args.batch_size, collate_fn=lambda b: collate_batch(b, tokenizer.pad_token_id), num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_corpus, batch_size=args.batch_size, collate_fn=lambda b: collate_batch(b, tokenizer.pad_token_id), num_workers=args.num_workers, pin_memory=True)

    model = StructXLMRoberta(model_name=args.model_path)
    model.to(device)

    print("Setting up optimizer with differential learning rates...")
    optimizer_grouped_parameters = [
        {"params": model.front_layers.parameters(), "lr": args.base_lr},
        {"params": model.embeddings.parameters(), "lr": args.base_lr},
        {"params": [p for n, p in model.rear_layers.named_parameters() if "struct_gate" not in n], "lr": args.base_lr},
        {"params": model.output_layer.parameters(), "lr": args.base_lr},
        {"params": model.parser.parameters(), "lr": args.parser_lr},
        {"params": [p for n, p in model.rear_layers.named_parameters() if "struct_gate" in n], "lr": args.parser_lr},
    ]
    optimizer = AdamW(optimizer_grouped_parameters)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=1000000)

    train(args, model, optimizer, scheduler, train_loader, valid_loader, device)

    print("\n--- Training Finished. ---")