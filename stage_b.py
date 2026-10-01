import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

import common as C

REDUCTION = 4
N_SLOTS = math.ceil(C.MAX_MEL_LEN / REDUCTION)
BATCH_SIZE_B = 4
GRAD_ACCUM_B = 8
LR_B = 1e-4
LR_HEAD = 1e-3
WEIGHT_DECAY_B = 1e-2
WARMUP_B = 500
MAX_STEPS_B = 20000
EVAL_EVERY_B = 1000
LOG_EVERY_B = 50
PATIENCE_B = 5
LR_FLOOR_B = 0.10
W_MEL = 1.0
W_STOP = 0.2
W_TEXT = 0.5
MEL_BIAS_INIT = -6.0
STOP_THRESHOLD = 0.5
MIN_VOCODE_FRAMES = 16
FP16 = True
GRAD_CKPT = True
TRAIN_ADAPTER = True
SEED = 1234
NUM_WORKERS = 2


class MelHead(nn.Module):
    def __init__(self, llm_dim=C.LLM_DIM, n_slots=N_SLOTS, reduction=REDUCTION, slot_std=0.02):
        super().__init__()
        self.reduction = reduction
        self.slots = nn.Parameter(torch.randn(n_slots, llm_dim) * slot_std)
        self.proj = nn.Linear(llm_dim, reduction * (C.N_MELS + 1))
        with torch.no_grad():
            bias = self.proj.bias.view(reduction, C.N_MELS + 1)
            bias[:, :C.N_MELS] = MEL_BIAS_INIT
            bias[:, C.N_MELS] = 0.0

    def forward(self, h):
        b, s, _ = h.shape
        out = self.proj(h.float()).view(b, s * self.reduction, C.N_MELS + 1)[:, :C.MAX_MEL_LEN]
        return out[..., :C.N_MELS].transpose(1, 2), out[..., C.N_MELS]


class Bundle:
    def __init__(self, tokenizer, llm, encoder, adapter, head, device, fp16):
        self.tokenizer = tokenizer
        self.llm = llm
        self.encoder = encoder
        self.adapter = adapter
        self.head = head
        self.device = device
        self.fp16 = fp16 and device.type == 'cuda'
        self.backbone, self.lm_head, self.embed = C.llm_parts(llm)
        self.eos_id = tokenizer.eos_token_id
        if self.eos_id is None:
            raise RuntimeError('tokenizer has no eos token')

    def train(self):
        self.llm.train()
        self.adapter.train(TRAIN_ADAPTER)
        self.head.train()
        self.encoder.eval()

    def eval(self):
        self.llm.eval()
        self.adapter.eval()
        self.head.eval()
        self.encoder.eval()

    def autocast(self):
        return torch.amp.autocast('cuda', enabled=self.fp16)

    def tokenize(self, text):
        ids = self.tokenizer(str(text), max_length=C.MAX_TEXT_LEN, truncation=True)['input_ids']
        return [int(i) for i in ids if int(i) != self.eos_id]


class StageBDataset(Dataset):
    def __init__(self, df, tokenize, audio_index=None):
        self.df = df.reset_index(drop=True)
        self.tokenize = tokenize
        self.audio_index = audio_index

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_row = row if self.audio_index is None else self.df.iloc[int(self.audio_index[idx])]
        try:
            wav = C.load_wav(row['wav_path'])
            audio_wav = wav if audio_row is row else C.load_wav(audio_row['wav_path'])
        except Exception as e:
            print(f'skipping {row["wav_path"]}: {e}', file=sys.stderr)
            return None
        voc = C.vocoder_mel(wav)
        return {
            'enc_mel': C.encoder_mel(audio_wav),
            'voc_mel': voc,
            'mel_len': voc.shape[1],
            'ids': self.tokenize(row['text']),
            'text': str(row['text']),
            'wav_path': str(row['wav_path']),
            'id': str(row['id']) if 'id' in row else Path(str(row['wav_path'])).stem,
        }


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    voc = torch.full((len(batch), C.N_MELS, C.MAX_MEL_LEN), C.LOG_FLOOR)
    for i, b in enumerate(batch):
        voc[i, :, :b['mel_len']] = b['voc_mel']
    return {
        'enc_mel': torch.stack([b['enc_mel'] for b in batch]),
        'voc_mel': voc,
        'mel_len': torch.tensor([b['mel_len'] for b in batch], dtype=torch.long),
        'ids': [b['ids'] for b in batch],
        'text': [b['text'] for b in batch],
        'wav_path': [b['wav_path'] for b in batch],
        'id': [b['id'] for b in batch],
    }


def make_loader(df, bundle, batch_size, shuffle, audio_index=None, workers=NUM_WORKERS):
    ds = StageBDataset(df, bundle.tokenize, audio_index)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      collate_fn=collate, pin_memory=bundle.device.type == 'cuda',
                      drop_last=False, persistent_workers=False)


def audio_tokens(bundle, enc_mel, audio_mode='real'):
    with torch.no_grad():
        z = bundle.encoder(enc_mel)
    tokens = bundle.adapter(z)
    if audio_mode == 'zero':
        tokens = torch.zeros_like(tokens)
    elif audio_mode != 'real':
        raise ValueError(audio_mode)
    return tokens


def build_sequence(bundle, tokens, ids_list):
    dtype = bundle.embed.weight.dtype
    device = tokens.device
    slots = bundle.head.slots.to(dtype)
    seqs, text_lens = [], []
    for i, ids in enumerate(ids_list):
        tok = torch.tensor(list(ids) + [bundle.eos_id], dtype=torch.long, device=device)
        seqs.append(torch.cat([tokens[i].to(dtype), bundle.embed(tok), slots], 0))
        text_lens.append(tok.numel())
    lens = torch.tensor([s.shape[0] for s in seqs], device=device)
    inputs = pad_sequence(seqs, batch_first=True)
    mask = (torch.arange(inputs.shape[1], device=device)[None] < lens[:, None]).long()
    return inputs, mask, torch.tensor(text_lens, device=device)


def gather(h, idx):
    return h.gather(1, idx.unsqueeze(-1).expand(-1, -1, h.shape[-1]))


def forward_b(bundle, enc_mel, ids_list, audio_mode='real'):
    tokens = audio_tokens(bundle, enc_mel, audio_mode)
    inputs, mask, text_lens = build_sequence(bundle, tokens, ids_list)
    h = bundle.backbone(inputs_embeds=inputs, attention_mask=mask, use_cache=False).last_hidden_state
    na = tokens.shape[1]
    n_slots = bundle.head.slots.shape[0]
    slot_idx = (na + text_lens)[:, None] + torch.arange(n_slots, device=h.device)[None]
    mel, stop = bundle.head(gather(h, slot_idx))
    return {'h': h, 'mel': mel, 'stop': stop, 'text_lens': text_lens, 'n_audio': na}


def text_ce(bundle, out, ids_list):
    h, text_lens, na = out['h'], out['text_lens'], out['n_audio']
    lmax = int(text_lens.max())
    idx = (na - 1 + torch.arange(lmax, device=h.device))[None].expand(h.shape[0], -1)
    labels = torch.full((h.shape[0], lmax), -100, dtype=torch.long, device=h.device)
    for i, ids in enumerate(ids_list):
        tgt = list(ids) + [bundle.eos_id]
        labels[i, :len(tgt)] = torch.tensor(tgt, device=h.device)
    logits = bundle.lm_head(gather(h, idx)).float()
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)


def frame_mask(mel_len, t):
    return torch.arange(t, device=mel_len.device)[None] < mel_len[:, None]


def mel_l1(pred, target, mel_len):
    mask = frame_mask(mel_len, pred.shape[-1]).float()
    per_frame = (pred.float() - target.float()).abs().mean(1)
    return (per_frame * mask).sum() / mask.sum().clamp(min=1.0)


def stop_bce(stop_logits, mel_len):
    target = (~frame_mask(mel_len, stop_logits.shape[-1])).float()
    return F.binary_cross_entropy_with_logits(stop_logits.float(), target)


def compute_losses(bundle, batch, audio_mode='real', with_text=True):
    enc = batch['enc_mel'].to(bundle.device, non_blocking=True)
    voc = batch['voc_mel'].to(bundle.device, non_blocking=True)
    mel_len = batch['mel_len'].to(bundle.device)
    with bundle.autocast():
        out = forward_b(bundle, enc, batch['ids'], audio_mode)
        l_mel = mel_l1(out['mel'], voc, mel_len)
        l_stop = stop_bce(out['stop'], mel_len)
        l_text = text_ce(bundle, out, batch['ids']) if with_text else torch.zeros((), device=bundle.device)
    total = W_MEL * l_mel + W_STOP * l_stop + W_TEXT * l_text
    return total, {'total': total.item(), 'mel_l1': l_mel.item(), 'stop_bce': l_stop.item(),
                   'text_ce': l_text.item()}, out


def predicted_lengths(stop_logits):
    done = torch.sigmoid(stop_logits.float()) > STOP_THRESHOLD
    any_done = done.any(1)
    first = done.float().argmax(1)
    lens = torch.where(any_done, first, torch.full_like(first, stop_logits.shape[1]))
    return lens.clamp(min=1)


@torch.no_grad()
def evaluate_losses(bundle, loader, audio_mode='real', max_batches=None):
    bundle.eval()
    sums, n = {}, 0
    for i, batch in enumerate(loader):
        if batch is None:
            continue
        if max_batches is not None and i >= max_batches:
            break
        _, d, _ = compute_losses(bundle, batch, audio_mode)
        if not all(math.isfinite(v) for v in d.values()):
            continue
        for k, v in d.items():
            sums[k] = sums.get(k, 0.0) + v
        n += 1
    if n == 0:
        raise RuntimeError('no finite validation batches')
    return {k: v / n for k, v in sums.items()}


@torch.no_grad()
def transcribe(bundle, enc_mel, max_new_tokens=C.MAX_TEXT_LEN):
    bundle.eval()
    with bundle.autocast():
        tokens = audio_tokens(bundle, enc_mel.to(bundle.device)).to(bundle.embed.weight.dtype)
        mask = torch.ones(tokens.shape[:2], dtype=torch.long, device=bundle.device)
        gen = bundle.llm.generate(inputs_embeds=tokens, attention_mask=mask, max_new_tokens=max_new_tokens,
                                  do_sample=False, num_beams=1, eos_token_id=bundle.eos_id,
                                  pad_token_id=bundle.eos_id)
    out = []
    for row in gen.tolist():
        ids = []
        for t in row:
            if t == bundle.eos_id:
                break
            ids.append(int(t))
        out.append(ids)
    return out


@torch.no_grad()
def predict_mels(bundle, enc_mel, ids_list, audio_mode='real'):
    bundle.eval()
    with bundle.autocast():
        out = forward_b(bundle, enc_mel.to(bundle.device), ids_list, audio_mode)
    return out['mel'].float(), predicted_lengths(out['stop'])


def load_vocoder(device, source=C.VOCODER_SOURCE, savedir=C.VOCODER_DIR):
    try:
        from speechbrain.inference.vocoders import HIFIGAN
    except ImportError:
        from speechbrain.pretrained import HIFIGAN
    return HIFIGAN.from_hparams(source=source, savedir=savedir, run_opts={'device': str(device)})


@torch.no_grad()
def vocode(vocoder, mels, lens):
    lens = lens.tolist() if torch.is_tensor(lens) else list(lens)
    t = max(1, max(int(n) for n in lens))
    mels = mels[:, :, :t].float()
    if t < MIN_VOCODE_FRAMES:
        mels = F.pad(mels, (0, MIN_VOCODE_FRAMES - t), value=C.LOG_FLOOR)
    wav = vocoder.decode_batch(mels)
    if wav.dim() == 3:
        wav = wav[:, 0]
    return [wav[i, :int(n) * C.HOP_LENGTH].float().cpu() for i, n in enumerate(lens)]


def save_wav(path, wav):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    C.write_audio(path, wav)


def build_bundle(device, stage_a_path=C.STAGE_A_CKPT, encoder_path=C.ENCODER_CKPT, require_stage_a=True,
                 fp16=FP16):
    tokenizer, llm = C.build_llm(device, fp16)
    encoder = C.load_frozen_encoder(device, encoder_path)
    adapter = C.ModalityAdapter().to(device)
    if stage_a_path and Path(stage_a_path).exists():
        C.load_stage_a(adapter, llm, device, stage_a_path)
        print(f'loaded Stage A weights from {stage_a_path}')
    elif require_stage_a:
        raise FileNotFoundError(f'Stage A checkpoint not found: {stage_a_path}')
    else:
        print('warning: Stage A checkpoint not loaded; adapter and LoRA start from scratch')
    emb_std = float(C.llm_parts(llm)[2].weight.float().std())
    head = MelHead(slot_std=emb_std if math.isfinite(emb_std) and emb_std > 0 else 0.02).to(device)
    for p in adapter.parameters():
        p.requires_grad = TRAIN_ADAPTER
    return Bundle(tokenizer, llm, encoder, adapter, head, device, fp16)


def trainable_state(bundle):
    return {
        'adapter_state': {k: v.detach().cpu() for k, v in bundle.adapter.state_dict().items()},
        'lora_state': C.lora_state_dict(bundle.llm),
        'head_state': {k: v.detach().cpu() for k, v in bundle.head.state_dict().items()},
        'config': stage_b_config(),
    }


def load_stage_b(bundle, path):
    ckpt = C.torch_load(path, bundle.device)
    for key in ('adapter_state', 'lora_state', 'head_state'):
        if key not in ckpt:
            raise RuntimeError(f'{path} missing {key}; not a Stage B checkpoint')
    cfg = ckpt.get('config', {})
    if cfg.get('reduction', REDUCTION) != REDUCTION or cfg.get('n_slots', N_SLOTS) != N_SLOTS:
        raise RuntimeError(f'{path} was trained with reduction={cfg.get("reduction")}, '
                           f'n_slots={cfg.get("n_slots")}; current {REDUCTION}, {N_SLOTS}')
    bundle.adapter.load_state_dict(ckpt['adapter_state'])
    C.load_lora_state(bundle.llm, ckpt['lora_state'])
    bundle.head.load_state_dict(ckpt['head_state'])
    return ckpt


def stage_b_config():
    return {
        'reduction': REDUCTION, 'n_slots': N_SLOTS, 'batch_size': BATCH_SIZE_B, 'grad_accum': GRAD_ACCUM_B,
        'lr': LR_B, 'lr_head': LR_HEAD, 'weight_decay': WEIGHT_DECAY_B, 'warmup': WARMUP_B,
        'max_steps': MAX_STEPS_B, 'eval_every': EVAL_EVERY_B, 'patience': PATIENCE_B,
        'lr_floor': LR_FLOOR_B, 'w_mel': W_MEL, 'w_stop': W_STOP, 'w_text': W_TEXT, 'fp16': FP16,
        'grad_ckpt': GRAD_CKPT, 'train_adapter': TRAIN_ADAPTER, 'seed': SEED, 'llm': C.LLM_ID,
        'vocoder': C.VOCODER_SOURCE, 'max_mel_len': C.MAX_MEL_LEN,
    }


def lr_lambda(max_steps):
    def f(step):
        if step < WARMUP_B:
            return (step + 1) / WARMUP_B
        progress = min(1.0, (step - WARMUP_B) / max(1, max_steps - WARMUP_B))
        return max(LR_FLOOR_B, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return f


def append_csv(path, row):
    new = not Path(path).exists()
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def atomic_save(obj, path):
    tmp = f'{path}.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(args):
    C.seed_everything(SEED)
    device = C.get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    C.ensure_splits()
    bundle = build_bundle(device, args.stage_a, args.encoder, require_stage_a=not args.allow_no_stage_a)
    if GRAD_CKPT and device.type == 'cuda':
        base = bundle.llm.get_base_model()
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        base.config.use_cache = False
    train_df = C.read_split(C.TRAIN_CSV, args.limit_train)
    val_df = C.read_split(C.VAL_CSV, args.limit_val)
    train_loader = make_loader(train_df, bundle, args.batch_size, True, workers=args.workers)
    val_loader = make_loader(val_df, bundle, args.batch_size, False, workers=args.workers)
    print(f'train {len(train_df)} | val {len(val_df)} | slots {N_SLOTS} x {REDUCTION} frames')

    lora_params = [p for p in bundle.llm.parameters() if p.requires_grad]
    adapter_params = [p for p in bundle.adapter.parameters() if p.requires_grad]
    head_params = list(bundle.head.parameters())
    if not lora_params:
        raise RuntimeError('no trainable LoRA parameters')
    groups = [{'params': lora_params + adapter_params, 'lr': LR_B},
              {'params': head_params, 'lr': LR_HEAD}]
    optimizer = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY_B, betas=(0.9, 0.98))
    max_steps = args.max_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(max_steps))
    scaler = torch.amp.GradScaler('cuda', enabled=bundle.fp16)
    print(f'trainable: LoRA {sum(p.numel() for p in lora_params) / 1e6:.2f}M | '
          f'adapter {sum(p.numel() for p in adapter_params) / 1e6:.2f}M | '
          f'mel head {sum(p.numel() for p in head_params) / 1e6:.2f}M')

    step, best_val, no_improve, best_step = 0, float('inf'), 0, -1
    last_path, best_path = out_dir / 'last.pt', out_dir / 'best.pt'
    metrics_path = out_dir / 'metrics.csv'
    if last_path.exists() and not args.fresh:
        ckpt = load_stage_b(bundle, last_path)
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        if 'scaler_state' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state'])
        step, best_val = ckpt['step'], ckpt['best_val']
        no_improve, best_step = ckpt['no_improve'], ckpt['best_step']
        print(f'resumed from {last_path} at step {step} (best {best_val:.4f} @ {best_step})')
    elif args.fresh and metrics_path.exists():
        metrics_path.rename(out_dir / f'metrics_{int(time.time())}.csv')

    if step == 0:
        base = evaluate_losses(bundle, val_loader, max_batches=args.max_val_batches)
        append_csv(metrics_path, {'step': 0, 'split': 'val', 'lr': 0.0, **base})
        print(f'step 0 val {json.dumps({k: round(v, 4) for k, v in base.items()})}')

    params = lora_params + adapter_params + head_params
    running, n_running, accum, skipped = {}, 0, 0, 0
    stop_training = step >= max_steps
    bundle.train()
    optimizer.zero_grad(set_to_none=True)
    while not stop_training:
        for batch in train_loader:
            if batch is None:
                continue
            loss, d, _ = compute_losses(bundle, batch)
            if not torch.isfinite(loss):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                print(f'non-finite loss at step {step}, skipping accumulation window ({skipped} total)')
                continue
            scaler.scale(loss / GRAD_ACCUM_B).backward()
            for k, v in d.items():
                running[k] = running.get(k, 0.0) + v
            n_running += 1
            accum += 1
            if accum < GRAD_ACCUM_B:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accum = 0
            step += 1

            if step % LOG_EVERY_B == 0 and n_running:
                avg = {k: v / n_running for k, v in running.items()}
                lr = scheduler.get_last_lr()[0]
                append_csv(metrics_path, {'step': step, 'split': 'train', 'lr': lr, **avg})
                print(f'step {step}/{max_steps} ' + ' '.join(f'{k}={v:.4f}' for k, v in avg.items())
                      + f' lr={lr:.2e}')
                running, n_running = {}, 0

            if step % args.eval_every == 0 or step >= max_steps:
                val = evaluate_losses(bundle, val_loader, max_batches=args.max_val_batches)
                append_csv(metrics_path, {'step': step, 'split': 'val', 'lr': scheduler.get_last_lr()[0], **val})
                print(f'VAL step {step} ' + ' '.join(f'{k}={v:.4f}' for k, v in val.items()))
                improved = val['total'] < best_val
                if improved:
                    best_val, best_step, no_improve = val['total'], step, 0
                    atomic_save({**trainable_state(bundle), 'step': step, 'val': val}, best_path)
                    print(f'  new best {best_val:.4f}')
                else:
                    no_improve += 1
                    print(f'  no improvement ({no_improve}/{PATIENCE_B})')
                atomic_save({**trainable_state(bundle), 'step': step, 'val': val,
                             'optimizer_state': optimizer.state_dict(),
                             'scheduler_state': scheduler.state_dict(),
                             'scaler_state': scaler.state_dict(), 'best_val': best_val,
                             'best_step': best_step, 'no_improve': no_improve}, last_path)
                bundle.train()
                if no_improve >= PATIENCE_B:
                    print('early stopping')
                    stop_training = True
            if step >= max_steps:
                stop_training = True
            if stop_training:
                break

    summary = {'best_step': best_step, 'best_val_total': best_val, 'final_step': step,
               'skipped_windows': skipped, 'train_utts': len(train_df), 'val_utts': len(val_df),
               'config': stage_b_config(), 'run_args': vars(args)}
    if best_path.exists():
        summary['best_val'] = C.torch_load(best_path, 'cpu').get('val')
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def infer(args):
    device = C.get_device()
    bundle = build_bundle(device, args.stage_a, args.encoder, require_stage_a=False)
    load_stage_b(bundle, args.ckpt)
    vocoder = load_vocoder(device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.text is not None and len(args.text) not in (0, len(args.wav)):
        raise ValueError('pass one --text per --wav, or none')
    for i, path in enumerate(args.wav):
        try:
            wav = C.load_wav(path)
        except Exception as e:
            print(f'skipping {path}: {e}', file=sys.stderr)
            continue
        enc = C.encoder_mel(wav)[None]
        if args.text:
            ids = [bundle.tokenize(args.text[i].lower().strip())]
        else:
            ids = transcribe(bundle, enc)
        text = bundle.tokenizer.decode(ids[0], skip_special_tokens=True)
        mel, lens = predict_mels(bundle, enc, ids)
        out = vocode(vocoder, mel, lens)[0]
        dst = out_dir / f'{Path(path).stem}_v2v.wav'
        save_wav(dst, out)
        print(f'{path} -> {dst} | {int(lens[0])} frames | text: {text}')


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--out_dir', default=C.STAGE_B_DIR)
    t.add_argument('--stage_a', default=C.STAGE_A_CKPT)
    t.add_argument('--encoder', default=C.ENCODER_CKPT)
    t.add_argument('--allow_no_stage_a', action='store_true')
    t.add_argument('--fresh', action='store_true')
    t.add_argument('--max_steps', type=int, default=MAX_STEPS_B)
    t.add_argument('--eval_every', type=int, default=EVAL_EVERY_B)
    t.add_argument('--batch_size', type=int, default=BATCH_SIZE_B)
    t.add_argument('--workers', type=int, default=NUM_WORKERS)
    t.add_argument('--limit_train', type=int, default=None)
    t.add_argument('--limit_val', type=int, default=None)
    t.add_argument('--max_val_batches', type=int, default=None)
    i = sub.add_parser('infer')
    i.add_argument('--wav', nargs='+', required=True)
    i.add_argument('--text', nargs='*', default=None)
    i.add_argument('--ckpt', default=os.path.join(C.STAGE_B_DIR, 'best.pt'))
    i.add_argument('--stage_a', default=C.STAGE_A_CKPT)
    i.add_argument('--encoder', default=C.ENCODER_CKPT)
    i.add_argument('--out_dir', default=os.path.join(C.STAGE_B_DIR, 'samples'))
    args = p.parse_args(argv)
    if args.cmd == 'train':
        if args.max_steps < 1 or args.eval_every < 1 or args.batch_size < 1:
            p.error('max_steps, eval_every and batch_size must be positive')
        train(args)
    else:
        infer(args)


if __name__ == '__main__':
    main()
