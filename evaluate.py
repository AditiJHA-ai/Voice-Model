import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import common as C
import stage_b as B

EVAL_SEED = 1234
ENC_BATCH = 16
MCD_COEFS = 24


def edit_distance(ref, hyp):
    if not ref:
        return len(hyp)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


class ErrorRate:
    def __init__(self):
        self.err = 0
        self.n = 0

    def add(self, ref, hyp):
        self.err += edit_distance(ref, hyp)
        self.n += len(ref)

    def value(self):
        return self.err / self.n if self.n else float('nan')


def simple_normalize(text):
    text = str(text).lower().replace('-', ' ')
    text = re.sub(r"[^a-z0-9' ]+", ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def derangement(n, seed):
    if n < 2:
        return None
    perm = np.random.default_rng(seed).permutation(n)
    idx = np.empty(n, dtype=np.int64)
    idx[perm] = np.roll(perm, -1)
    return idx


def stats(values):
    v = np.array([x for x in values if x is not None and math.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {'mean': None, 'std': None, 'n': 0}
    return {'mean': float(v.mean()), 'std': float(v.std(ddof=1)) if v.size > 1 else 0.0, 'n': int(v.size)}


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
    print(json.dumps(obj, indent=2))


class EncoderEvalDataset(Dataset):
    def __init__(self, df, seed):
        self.df = df
        self.seed = seed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            wav = C.load_wav(row['wav_path'])
        except Exception as e:
            print(f'skipping {row["wav_path"]}: {e}', file=sys.stderr)
            return None
        mel = C.encoder_mel(wav)
        g = torch.Generator().manual_seed(self.seed + idx)
        ids = C.text_to_ids(row['text'])
        if not ids:
            return None
        return mel, C.augment_mel(mel, g), torch.tensor(ids), C.n_mel_frames(wav.shape[1]), str(row['text'])


def enc_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    mel, aug, ids, frames, text = zip(*batch)
    return (torch.stack(mel), torch.stack(aug),
            torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0),
            torch.tensor([len(i) for i in ids]), torch.tensor(frames), list(text))


def info_nce(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    sim = a @ b.T / C.TEMP
    lab = torch.arange(a.shape[0], device=a.device)
    return (F.cross_entropy(sim, lab) + F.cross_entropy(sim.T, lab)) / 2


def masked_mean(z, n_valid):
    mask = (torch.arange(z.shape[1], device=z.device)[None] < n_valid[:, None]).float()
    return (z * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)


def greedy_ctc(log_probs, n_valid):
    out = []
    for lp, n in zip(log_probs.argmax(-1).tolist(), n_valid.tolist()):
        seq, prev = [], 0
        for t in lp[:n]:
            if t != prev and t != 0:
                seq.append(t)
            prev = t
        out.append(C.ids_to_text(seq))
    return out


def cosine_summary(pooled, pooled_aug):
    p = F.normalize(pooled, dim=-1)
    pa = F.normalize(pooled_aug, dim=-1)
    sim = p @ p.T
    n = sim.shape[0]
    off = sim[~torch.eye(n, dtype=torch.bool)]
    inv = (p * pa).sum(-1)
    return {
        'cross_utterance_cosine': stats(off.tolist()),
        'augmentation_invariance_cosine': stats(inv.tolist()),
    }


@torch.no_grad()
def eval_encoder(args):
    C.seed_everything(EVAL_SEED)
    device = C.get_device()
    C.ensure_splits()
    ckpt = C.load_encoder_checkpoint(args.encoder, device)
    missing = [k for k in ('decoder_state', 'ctc_head_state') if k not in ckpt]
    if missing:
        raise RuntimeError(f'{args.encoder} lacks {missing}; Table I needs the full encoder checkpoint')
    enc = C.CustomSpeechEncoder().to(device).eval()
    dec = C.ReconDecoder().to(device).eval()
    ctc = C.CTCHead().to(device).eval()
    enc.load_state_dict(ckpt['encoder_state'])
    dec.load_state_dict(ckpt['decoder_state'])
    ctc.load_state_dict(ckpt['ctc_head_state'])
    df = C.read_split(args.split_csv or C.VAL_CSV, args.limit)
    loader = DataLoader(EncoderEvalDataset(df, EVAL_SEED), batch_size=ENC_BATCH, shuffle=False,
                        num_workers=args.workers, collate_fn=enc_collate)
    ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    sums = {k: 0.0 for k in ('total', 'recon', 'ctc', 'contrast', 'recon_masked', 'ctc_true_len',
                             'contrast_masked')}
    n_batches = 0
    cer = ErrorRate()
    wer = ErrorRate()
    pools = {'padded': [], 'padded_aug': [], 'masked': [], 'masked_aug': []}
    rows = []
    for batch in loader:
        if batch is None:
            continue
        mel, aug, ids, ids_len, frames, texts = batch
        mel, aug, ids, ids_len, frames = (x.to(device) for x in (mel, aug, ids, ids_len, frames))
        b = mel.shape[0]
        with torch.amp.autocast('cuda', enabled=args.fp16 and device.type == 'cuda'):
            z = enc(mel)
            za = enc(aug)
            mel_hat = dec(z)
            lp = ctc(z).float().log_softmax(-1)
        z, za, mel_hat = z.float(), za.float(), mel_hat.float()
        t = min(mel_hat.shape[-1], mel.shape[-1])
        recon = F.mse_loss(mel_hat[..., :t], mel[..., :t])
        fmask = (torch.arange(t, device=device)[None] < frames[:, None]).float()
        recon_m = (((mel_hat[..., :t] - mel[..., :t]) ** 2).mean(1) * fmask).sum() / fmask.sum()
        t_enc = z.shape[1]
        n_valid = ((frames + 1) // 2).clamp(max=t_enc)
        l_ctc = ctc_loss(lp.permute(1, 0, 2), ids, torch.full((b,), t_enc, device=device, dtype=torch.long), ids_len)
        l_ctc_true = ctc_loss(lp.permute(1, 0, 2), ids, n_valid, ids_len)
        pz, pza = z.mean(1), za.mean(1)
        mz, mza = masked_mean(z, n_valid), masked_mean(za, n_valid)
        l_con = info_nce(pz, pza) if b > 1 else torch.tensor(float('nan'))
        l_con_m = info_nce(mz, mza) if b > 1 else torch.tensor(float('nan'))
        total = C.W_RECON * recon + C.W_CTC * l_ctc + C.W_CONTRAST * l_con
        vals = {'total': total, 'recon': recon, 'ctc': l_ctc, 'contrast': l_con, 'recon_masked': recon_m,
                'ctc_true_len': l_ctc_true, 'contrast_masked': l_con_m}
        if not all(math.isfinite(float(v)) for v in vals.values()):
            continue
        for k, v in vals.items():
            sums[k] += float(v)
        n_batches += 1
        for name, val in (('padded', pz), ('padded_aug', pza), ('masked', mz), ('masked_aug', mza)):
            pools[name].append(val.cpu())
        for hyp, ref in zip(greedy_ctc(lp, n_valid), texts):
            ref_n = ''.join(c for c in ref.lower() if c in C.VOCAB)
            cer.add(list(ref_n), list(hyp))
            wer.add(ref_n.split(), hyp.split())
            if len(rows) < args.n_examples:
                rows.append({'ref': ref_n, 'hyp': hyp})
    if n_batches == 0:
        raise RuntimeError('no valid batches')
    losses = {k: v / n_batches for k, v in sums.items()}
    result = {
        'checkpoint': args.encoder,
        'checkpoint_step': ckpt.get('step'),
        'checkpoint_val_loss_logged': ckpt.get('val_loss'),
        'split': args.split_csv or C.VAL_CSV,
        'utterances': sum(p.shape[0] for p in pools['padded']),
        'batch_size': ENC_BATCH,
        'losses_as_trained': {k: losses[k] for k in ('total', 'recon', 'ctc', 'contrast')},
        'losses_padding_masked': {k: losses[k] for k in ('recon_masked', 'ctc_true_len', 'contrast_masked')},
        'ctc_greedy_cer': cer.value(),
        'ctc_greedy_wer': wer.value(),
        'geometry_padded_meanpool': cosine_summary(torch.cat(pools['padded']), torch.cat(pools['padded_aug'])),
        'geometry_masked_meanpool': cosine_summary(torch.cat(pools['masked']), torch.cat(pools['masked_aug'])),
        'examples': rows,
        'seed': EVAL_SEED,
        'fp16': bool(args.fp16 and device.type == 'cuda'),
    }
    write_json(Path(args.out_dir) / 'encoder_metrics.json', result)


class StageADataset(Dataset):
    def __init__(self, df, tokenizer, audio_index=None):
        self.df = df
        self.tokenizer = tokenizer
        self.audio_index = audio_index

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src = self.df.iloc[int(self.audio_index[idx])] if self.audio_index is not None else row
        try:
            mel = C.encoder_mel(C.load_wav(src['wav_path']))
        except Exception as e:
            print(f'skipping {src["wav_path"]}: {e}', file=sys.stderr)
            return None
        tok = self.tokenizer(str(row['text']), max_length=C.MAX_TEXT_LEN, truncation=True,
                             padding='max_length', return_tensors='pt')
        return mel, tok['input_ids'][0], tok['attention_mask'][0]


def a_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return tuple(torch.stack(x) for x in zip(*batch))


@torch.no_grad()
def eval_stage_a(args):
    C.seed_everything(EVAL_SEED)
    device = C.get_device()
    C.ensure_splits()
    tokenizer, llm = C.build_llm(device, args.fp16)
    encoder = C.load_frozen_encoder(device, args.encoder)
    adapter = C.ModalityAdapter().to(device)
    ckpt = C.load_stage_a(adapter, llm, device, args.stage_a)
    llm.eval()
    adapter.eval()
    df = C.read_split(args.split_csv or C.VAL_CSV, args.limit)
    perm = derangement(len(df), EVAL_SEED)
    results = {}
    for mode in ('real', 'shuffled', 'zero'):
        if mode == 'shuffled' and perm is None:
            continue
        ds = StageADataset(df, tokenizer, perm if mode == 'shuffled' else None)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            collate_fn=a_collate)
        total, n = 0.0, 0
        for batch in loader:
            if batch is None:
                continue
            mel, ids, mask = (x.to(device) for x in batch)
            b = mel.shape[0]
            with torch.amp.autocast('cuda', enabled=args.fp16 and device.type == 'cuda'):
                tokens = adapter(encoder(mel))
                if mode == 'zero':
                    tokens = torch.zeros_like(tokens)
                emb = llm.get_input_embeddings()(ids)
                inputs = torch.cat([tokens.to(emb.dtype), emb], 1)
                full_mask = torch.cat([torch.ones(b, tokens.shape[1], dtype=mask.dtype, device=device), mask], 1)
                labels = torch.cat([torch.full((b, tokens.shape[1]), -100, dtype=ids.dtype, device=device), ids], 1)
                labels[labels == tokenizer.pad_token_id] = -100
                loss = llm(inputs_embeds=inputs, attention_mask=full_mask, labels=labels).loss
            if math.isfinite(loss.item()):
                total += loss.item()
                n += 1
        results[mode] = total / n if n else None
        print(f'{mode}: {results[mode]}')
    write_json(Path(args.out_dir) / 'stage_a_control.json', {
        'checkpoint': args.stage_a, 'checkpoint_step': ckpt.get('step'),
        'checkpoint_val_loss_logged': ckpt.get('val_loss'), 'split': args.split_csv or C.VAL_CSV,
        'utterances': len(df), 'text_ce': results,
        'note': 'shuffled = audio tokens from a different utterance (fixed derangement); zero = all-zero audio tokens',
        'seed': EVAL_SEED,
    })


def load_asr(name, device, fp16):
    from transformers import pipeline
    kwargs = {'model': name, 'device': 0 if device.type == 'cuda' else -1}
    if device.type == 'cuda' and fp16:
        kwargs['dtype'] = torch.float16
    pipe = pipeline('automatic-speech-recognition', **kwargs)
    english_only = name.endswith('.en')
    try:
        pipe.tokenizer.normalize('test')
        norm = pipe.tokenizer.normalize
    except Exception:
        norm = simple_normalize

    def run(wavs):
        if not wavs:
            return []
        inputs = [{'raw': w.numpy().astype(np.float32), 'sampling_rate': C.MEL_SR} for w in wavs]
        gk = {} if english_only else {'language': 'english', 'task': 'transcribe'}
        outs = pipe(inputs, batch_size=len(inputs), generate_kwargs=gk)
        return [o['text'] for o in outs]

    return run, norm


def mcd_dtw(ref_wav, syn_wav, n_coef=MCD_COEFS):
    from scipy.fft import dct
    from scipy.spatial.distance import cdist
    import librosa
    if ref_wav.numel() < C.HOP_LENGTH or syn_wav.numel() < C.HOP_LENGTH:
        return None
    r = C.vocoder_mel(ref_wav[None].float()).T.numpy().astype(np.float64)
    s = C.vocoder_mel(syn_wav[None].float()).T.numpy().astype(np.float64)
    cr = dct(r, type=2, norm='ortho', axis=1)[:, 1:n_coef + 1]
    cs = dct(s, type=2, norm='ortho', axis=1)[:, 1:n_coef + 1]
    _, wp = librosa.sequence.dtw(C=cdist(cr, cs))
    diff = cr[wp[:, 0]] - cs[wp[:, 1]]
    return float((10.0 / math.log(10.0)) * math.sqrt(2.0) * np.mean(np.sqrt((diff ** 2).sum(1))))


@torch.no_grad()
def eval_stage_b(args):
    C.seed_everything(EVAL_SEED)
    device = C.get_device()
    C.ensure_splits()
    bundle = B.build_bundle(device, args.stage_a, args.encoder, require_stage_a=False, fp16=args.fp16)
    ckpt = B.load_stage_b(bundle, args.ckpt)
    bundle.eval()
    vocoder = B.load_vocoder(device)
    asr, norm = load_asr(args.asr_model, device, args.fp16)
    df = C.read_split(args.split_csv or C.TEST_CSV, args.limit)
    out_dir = Path(args.out_dir)
    sample_dir = out_dir / 'stage_b_samples'
    conditions = ['ground_truth', 'copy_synthesis', 'stage_b_gt_text', 'stage_b_v2v']
    rows = []
    llm_cer, llm_wer = ErrorRate(), ErrorRate()
    judge = {c: (ErrorRate(), ErrorRate()) for c in conditions}
    loader = B.make_loader(df, bundle, args.batch_size, False, workers=args.workers)
    tf_l1, tf_len_err = [], []
    saved = 0
    for batch in loader:
        if batch is None:
            continue
        enc = batch['enc_mel'].to(device)
        voc = batch['voc_mel'].to(device)
        mel_len = batch['mel_len'].to(device)
        refs = [C.load_wav(p)[0] for p in batch['wav_path']]
        with bundle.autocast():
            out = B.forward_b(bundle, enc, batch['ids'])
        tf_l1.extend(per_utt_l1(out['mel'].float(), voc, mel_len))
        tf_len_err.extend((B.predicted_lengths(out['stop']) - mel_len).abs().float().tolist())
        copy = B.vocode(vocoder, voc, mel_len)
        mel_gt, len_gt = B.predict_mels(bundle, enc, batch['ids'])
        wav_gt = B.vocode(vocoder, mel_gt, len_gt)
        hyp_ids = B.transcribe(bundle, enc)
        hyp_text = [bundle.tokenizer.decode(i, skip_special_tokens=True) for i in hyp_ids]
        mel_v, len_v = B.predict_mels(bundle, enc, hyp_ids)
        wav_v = B.vocode(vocoder, mel_v, len_v)
        wavs = {'ground_truth': refs, 'copy_synthesis': copy, 'stage_b_gt_text': wav_gt, 'stage_b_v2v': wav_v}
        heard = {c: asr(wavs[c]) for c in conditions}
        for i, ref_text in enumerate(batch['text']):
            ref_n = norm(ref_text)
            llm_n = norm(hyp_text[i])
            llm_cer.add(list(ref_n), list(llm_n))
            llm_wer.add(ref_n.split(), llm_n.split())
            row = {'id': batch['id'][i], 'ref_text': ref_text, 'llm_transcript': hyp_text[i],
                   'true_frames': int(mel_len[i]), 'tf_mel_l1': tf_l1[len(rows)],
                   'frames_gt_text': int(len_gt[i]), 'frames_v2v': int(len_v[i])}
            for c in conditions:
                h = norm(heard[c][i])
                judge[c][0].add(ref_n.split(), h.split())
                judge[c][1].add(list(ref_n), list(h))
                row[f'asr_{c}'] = heard[c][i]
                if c != 'ground_truth':
                    row[f'mcd_{c}'] = mcd_dtw(refs[i], wavs[c][i])
            rows.append(row)
            if saved < args.n_samples:
                for c in conditions:
                    B.save_wav(sample_dir / f'{batch["id"][i]}_{c}.wav', wavs[c][i])
                saved += 1
        print(f'{len(rows)}/{len(df)} utterances')
    if not rows:
        raise RuntimeError('no utterances evaluated')

    controls = {}
    for mode in ('shuffled', 'zero'):
        perm = derangement(len(df), EVAL_SEED) if mode == 'shuffled' else None
        if mode == 'shuffled' and perm is None:
            continue
        l1s, errs = [], []
        ldr = B.make_loader(df, bundle, args.batch_size, False, audio_index=perm, workers=args.workers)
        for batch in ldr:
            if batch is None:
                continue
            enc = batch['enc_mel'].to(device)
            voc = batch['voc_mel'].to(device)
            mel_len = batch['mel_len'].to(device)
            with bundle.autocast():
                out = B.forward_b(bundle, enc, batch['ids'], 'zero' if mode == 'zero' else 'real')
            l1s.extend(per_utt_l1(out['mel'].float(), voc, mel_len))
            errs.extend((B.predicted_lengths(out['stop']) - mel_len).abs().float().tolist())
        controls[mode] = {'mel_l1': stats(l1s), 'abs_length_error_frames': stats(errs)}

    with open(out_dir / 'stage_b_per_utterance.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {
        'checkpoint': args.ckpt, 'checkpoint_step': ckpt.get('step'), 'checkpoint_val': ckpt.get('val'),
        'split': args.split_csv or C.TEST_CSV, 'utterances': len(rows), 'asr_judge': args.asr_model,
        'vocoder': C.VOCODER_SOURCE,
        'llm_transcript': {'wer': llm_wer.value(), 'cer': llm_cer.value()},
        'judge_wer': {c: judge[c][0].value() for c in conditions},
        'judge_cer': {c: judge[c][1].value() for c in conditions},
        'mcd_dtw_db': {c: stats([r[f'mcd_{c}'] for r in rows]) for c in conditions if c != 'ground_truth'},
        'teacher_forced': {
            'real': {'mel_l1': stats(tf_l1), 'abs_length_error_frames': stats(tf_len_err)},
            **controls,
        },
        'definitions': {
            'mcd_dtw_db': f'(10/ln10)*sqrt(2)*mean ||c_ref-c_syn|| over DTW path; c = DCT-II(ortho) coefficients '
                          f'1..{MCD_COEFS} of the 80-band natural-log Slaney mel spectrogram, 22.05 kHz, hop 256',
            'judge_wer': 'ASR judge transcript vs reference, both passed through the judge tokenizer normalizer',
            'llm_transcript': 'greedy LLM transcript from audio tokens vs reference',
            'teacher_forced': 'reference text given; mel L1 over true frames; shuffled = audio from a different '
                              'utterance via fixed derangement; zero = all-zero audio tokens',
        },
        'seed': EVAL_SEED,
    }
    write_json(out_dir / 'stage_b_metrics.json', summary)


def per_utt_l1(pred, target, mel_len):
    mask = B.frame_mask(mel_len, pred.shape[-1]).float()
    per = ((pred - target.float()).abs().mean(1) * mask).sum(1) / mask.sum(1).clamp(min=1.0)
    return per.tolist()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['encoder', 'stage_a', 'stage_b'])
    p.add_argument('--encoder', default=C.ENCODER_CKPT)
    p.add_argument('--stage_a', default=C.STAGE_A_CKPT)
    p.add_argument('--ckpt', default=os.path.join(C.STAGE_B_DIR, 'best.pt'))
    p.add_argument('--split_csv', default=None)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--out_dir', default=C.EVAL_DIR)
    p.add_argument('--asr_model', default='openai/whisper-large-v3-turbo')
    p.add_argument('--n_samples', type=int, default=10)
    p.add_argument('--n_examples', type=int, default=10)
    p.add_argument('--no_fp16', dest='fp16', action='store_false')
    args = p.parse_args(argv)
    if args.batch_size < 1:
        p.error('batch_size must be positive')
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    {'encoder': eval_encoder, 'stage_a': eval_stage_a, 'stage_b': eval_stage_b}[args.mode](args)


if __name__ == '__main__':
    main()
