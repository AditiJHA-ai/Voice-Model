import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import common as C

BATCH_SIZE_A = 8
GRAD_ACCUM_A = 4
LR_A = 2e-4
WEIGHT_DECAY_A = 1e-2
WARMUP_A = 500
MAX_STEPS_A = 15000
EVAL_EVERY_A = 1500
LOG_EVERY_A = 50
PATIENCE_A = 4
LR_FLOOR_A = 0.10
FP16 = True
SEED = 1234
NUM_WORKERS = 4
LIBRI_VAL_FRACTION = 0.02


class StageADataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            mel = C.encoder_mel(C.load_wav(row['wav_path']))
        except Exception as e:
            print(f'skipping {row["wav_path"]}: {e}', file=sys.stderr)
            return None
        tok = self.tokenizer(str(row['text']), max_length=C.MAX_TEXT_LEN, truncation=True,
                             padding='max_length', return_tensors='pt')
        return mel, tok['input_ids'][0], tok['attention_mask'][0]


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return tuple(torch.stack(x) for x in zip(*batch))


def librispeech_manifest(root):
    base = Path(root) / 'train-clean-100'
    if not base.exists():
        raise FileNotFoundError(f'{base} not found')
    records = []
    for trans in sorted(base.rglob('*.trans.txt')):
        with open(trans) as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) != 2:
                    continue
                wav = trans.parent / f'{parts[0]}.flac'
                text = parts[1].lower().strip()
                if wav.exists() and 5 <= len(text) <= 190:
                    records.append({'id': parts[0], 'wav_path': str(wav), 'text': text})
    if not records:
        raise RuntimeError(f'no LibriSpeech utterances under {base}')
    return pd.DataFrame(records)


def build_splits(args):
    train = C.read_split(C.TRAIN_CSV, args.limit_train)
    val = C.read_split(C.VAL_CSV, args.limit_val)
    if args.librispeech:
        ls = librispeech_manifest(args.librispeech).sample(frac=1.0, random_state=42).reset_index(drop=True)
        n_val = max(1, int(len(ls) * LIBRI_VAL_FRACTION))
        train = pd.concat([train, ls.iloc[n_val:]], ignore_index=True)
        val = pd.concat([val, ls.iloc[:n_val]], ignore_index=True)
    return train, val


def stage_a_loss(tokenizer, llm, encoder, adapter, mel, ids, mask):
    b = mel.shape[0]
    with torch.no_grad():
        z = encoder(mel)
    tokens = adapter(z)
    emb = llm.get_input_embeddings()(ids)
    inputs = torch.cat([tokens.to(emb.dtype), emb], 1)
    n = tokens.shape[1]
    full_mask = torch.cat([torch.ones(b, n, dtype=mask.dtype, device=mask.device), mask], 1)
    labels = torch.cat([torch.full((b, n), -100, dtype=ids.dtype, device=ids.device), ids], 1)
    labels[labels == tokenizer.pad_token_id] = -100
    return llm(inputs_embeds=inputs, attention_mask=full_mask, labels=labels).loss


@torch.no_grad()
def validate(tokenizer, llm, encoder, adapter, loader, device, fp16):
    llm.eval()
    adapter.eval()
    total, n = 0.0, 0
    for batch in loader:
        if batch is None:
            continue
        mel, ids, mask = (x.to(device, non_blocking=True) for x in batch)
        with torch.amp.autocast('cuda', enabled=fp16):
            loss = stage_a_loss(tokenizer, llm, encoder, adapter, mel, ids, mask)
        if math.isfinite(loss.item()):
            total += loss.item()
            n += 1
    if n == 0:
        raise RuntimeError('no finite validation batches')
    return total / n


def lr_lambda(max_steps):
    def f(step):
        if step < WARMUP_A:
            return (step + 1) / WARMUP_A
        progress = min(1.0, (step - WARMUP_A) / max(1, max_steps - WARMUP_A))
        return max(LR_FLOOR_A, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return f


def atomic_save(obj, path):
    tmp = f'{path}.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def append_csv(path, step, split, lr, loss):
    new = not Path(path).exists()
    with open(path, 'a') as f:
        if new:
            f.write('step,split,lr,loss\n')
        f.write(f'{step},{split},{lr},{loss}\n')


def train(args):
    C.seed_everything(SEED)
    device = C.get_device()
    fp16 = FP16 and device.type == 'cuda'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    C.ensure_splits()
    tokenizer, llm = C.build_llm(device, FP16)
    encoder = C.load_frozen_encoder(device, args.encoder)
    adapter = C.ModalityAdapter().to(device)
    train_df, val_df = build_splits(args)
    train_loader = DataLoader(StageADataset(train_df, tokenizer), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(StageADataset(val_df, tokenizer), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=collate, pin_memory=device.type == 'cuda')
    print(f'train {len(train_df)} | val {len(val_df)}')
    params = list(adapter.parameters()) + [p for p in llm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR_A, weight_decay=WEIGHT_DECAY_A, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(args.max_steps))
    scaler = torch.amp.GradScaler('cuda', enabled=fp16)

    step, best_val, best_step, no_improve = 0, float('inf'), -1, 0
    last_path, best_path, metrics = out_dir / 'last.pt', out_dir / 'best.pt', out_dir / 'metrics.csv'
    if last_path.exists() and not args.fresh:
        ckpt = C.torch_load(last_path, device)
        adapter.load_state_dict(ckpt['adapter_state'])
        C.load_lora_state(llm, ckpt['lora_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        scaler.load_state_dict(ckpt['scaler_state'])
        step, best_val, best_step, no_improve = ckpt['step'], ckpt['best_val'], ckpt['best_step'], ckpt['no_improve']
        print(f'resumed at step {step} (best {best_val:.4f} @ {best_step})')
    elif args.fresh and metrics.exists():
        metrics.rename(out_dir / f'metrics_{int(time.time())}.csv')

    if step == 0:
        base = validate(tokenizer, llm, encoder, adapter, val_loader, device, fp16)
        append_csv(metrics, 0, 'val', 0.0, base)
        print(f'step 0 val loss {base:.4f}')

    adapter.train()
    llm.train()
    optimizer.zero_grad(set_to_none=True)
    running, n_running, accum, skipped = 0.0, 0, 0, 0
    done = step >= args.max_steps
    while not done:
        for batch in train_loader:
            if batch is None:
                continue
            mel, ids, mask = (x.to(device, non_blocking=True) for x in batch)
            with torch.amp.autocast('cuda', enabled=fp16):
                loss = stage_a_loss(tokenizer, llm, encoder, adapter, mel, ids, mask)
            if not torch.isfinite(loss):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                continue
            scaler.scale(loss / GRAD_ACCUM_A).backward()
            running += loss.item()
            n_running += 1
            accum += 1
            if accum < GRAD_ACCUM_A:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accum = 0
            step += 1
            lr = scheduler.get_last_lr()[0]
            if step % LOG_EVERY_A == 0 and n_running:
                append_csv(metrics, step, 'train', lr, running / n_running)
                print(f'step {step}/{args.max_steps} loss={running / n_running:.4f} lr={lr:.2e}')
                running, n_running = 0.0, 0
            if step % args.eval_every == 0 or step >= args.max_steps:
                val = validate(tokenizer, llm, encoder, adapter, val_loader, device, fp16)
                append_csv(metrics, step, 'val', lr, val)
                print(f'VAL step {step} loss={val:.4f}')
                state = {'step': step, 'val_loss': val, 'adapter_state': adapter.state_dict(),
                         'lora_state': C.lora_state_dict(llm)}
                if val < best_val:
                    best_val, best_step, no_improve = val, step, 0
                    atomic_save(state, best_path)
                    print('  new best')
                else:
                    no_improve += 1
                    print(f'  no improvement ({no_improve}/{PATIENCE_A})')
                atomic_save({**state, 'optimizer_state': optimizer.state_dict(),
                             'scheduler_state': scheduler.state_dict(), 'scaler_state': scaler.state_dict(),
                             'best_val': best_val, 'best_step': best_step, 'no_improve': no_improve}, last_path)
                adapter.train()
                llm.train()
                if no_improve >= PATIENCE_A:
                    print('early stopping')
                    done = True
            if step >= args.max_steps:
                done = True
            if done:
                break

    summary = {'best_step': best_step, 'best_val_loss': best_val, 'final_step': step, 'skipped_batches': skipped,
               'train_utts': len(train_df), 'val_utts': len(val_df), 'librispeech': bool(args.librispeech),
               'config': {'batch_size': args.batch_size, 'grad_accum': GRAD_ACCUM_A, 'lr': LR_A,
                          'warmup': WARMUP_A, 'max_steps': args.max_steps, 'eval_every': args.eval_every,
                          'patience': PATIENCE_A, 'lr_floor': LR_FLOOR_A, 'weight_decay': WEIGHT_DECAY_A,
                          'fp16': fp16, 'seed': SEED, 'llm': C.LLM_ID}}
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default=C.STAGE_A_DIR)
    p.add_argument('--encoder', default=C.ENCODER_CKPT)
    p.add_argument('--librispeech', default=None, help='LibriSpeech root containing train-clean-100')
    p.add_argument('--fresh', action='store_true')
    p.add_argument('--max_steps', type=int, default=MAX_STEPS_A)
    p.add_argument('--eval_every', type=int, default=EVAL_EVERY_A)
    p.add_argument('--batch_size', type=int, default=BATCH_SIZE_A)
    p.add_argument('--workers', type=int, default=NUM_WORKERS)
    p.add_argument('--limit_train', type=int, default=None)
    p.add_argument('--limit_val', type=int, default=None)
    args = p.parse_args(argv)
    if args.max_steps < 1 or args.eval_every < 1 or args.batch_size < 1:
        p.error('max_steps, eval_every and batch_size must be positive')
    train(args)


if __name__ == '__main__':
    main()
