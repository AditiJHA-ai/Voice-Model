import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

DRIVE_DIR = os.environ.get('VM_DRIVE_DIR', '/content/drive/MyDrive')
DATA_ROOT = os.environ.get('VM_DATA_ROOT', 'LJSpeech-1.1')
TRAIN_CSV = os.environ.get('VM_TRAIN_CSV', 'train.csv')
VAL_CSV = os.environ.get('VM_VAL_CSV', 'val.csv')
TEST_CSV = os.environ.get('VM_TEST_CSV', 'test.csv')
ENCODER_CKPT = os.path.join(DRIVE_DIR, 'encoder_best_step12000.pt')
STAGE_A_DIR = os.path.join(DRIVE_DIR, 'stage_a')
STAGE_A_CKPT = os.path.join(STAGE_A_DIR, 'best.pt')
STAGE_B_DIR = os.path.join(DRIVE_DIR, 'stage_b')
EVAL_DIR = os.path.join(DRIVE_DIR, 'eval')
VOCODER_SOURCE = os.environ.get('VM_VOCODER', 'speechbrain/tts-hifigan-ljspeech')
VOCODER_DIR = os.path.join(DRIVE_DIR, 'pretrained', 'tts-hifigan-ljspeech')

MEL_SR = 22050
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
FMIN = 0.0
FMAX = 8000.0
MAX_DURATION = 10.0
MAX_MEL_LEN = int(MAX_DURATION * MEL_SR / HOP_LENGTH)
MAX_SAMPLES = MAX_MEL_LEN * HOP_LENGTH
LOG_FLOOR = math.log(1e-5)

HIDDEN_DIM = 768
ENCODER_HEADS = 12
ENCODER_LAYERS = 6
FFN_DIM = 3072
DROPOUT = 0.1

CHARS = list(" abcdefghijklmnopqrstuvwxyz',-.")
VOCAB = {c: i + 1 for i, c in enumerate(CHARS)}
IDX2CHAR = {i: c for c, i in VOCAB.items()}
VOCAB_SIZE = len(CHARS) + 1

W_RECON = 0.4
W_CTC = 0.4
W_CONTRAST = 0.2
TEMP = 0.07

LLM_ID = os.environ.get('VM_LLM_ID', 'HuggingFaceTB/SmolLM2-1.7B')
LLM_DIM = 2048
NUM_AUDIO_TOKENS = 32
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ['q_proj', 'v_proj', 'o_proj']
MAX_TEXT_LEN = 128


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ensure_splits(seed=42):
    if Path(TRAIN_CSV).exists() and Path(VAL_CSV).exists() and Path(TEST_CSV).exists():
        return
    meta = Path(DATA_ROOT) / 'metadata.csv'
    if not meta.exists():
        raise FileNotFoundError(f'{meta} not found; download LJSpeech-1.1 first')
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(meta, sep='|', header=None, names=['id', 'transcript', 'normalized'])
    df['wav_path'] = f'{DATA_ROOT}/wavs/' + df['id'] + '.wav'
    df['text'] = df['normalized'].str.lower().str.strip()
    df['text_len'] = df['text'].str.len()
    df = df[(df['text_len'] >= 5) & (df['text_len'] <= 190)].copy()
    train_val, test = train_test_split(df, test_size=0.10, random_state=seed)
    train, val = train_test_split(train_val, test_size=0.111, random_state=seed)
    cols = ['id', 'wav_path', 'text']
    for split, path in ((train, TRAIN_CSV), (val, VAL_CSV), (test, TEST_CSV)):
        if Path(path).exists():
            old = set(pd.read_csv(path)['wav_path'])
            new = set(split['wav_path'])
            if old != new:
                print(f'warning: existing {path} differs from regenerated split '
                      f'({len(old ^ new)} paths differ); keeping existing file')
            continue
        split[cols].to_csv(path, index=False)
    overlap = set(pd.read_csv(TEST_CSV)['wav_path']) & (
        set(pd.read_csv(TRAIN_CSV)['wav_path']) | set(pd.read_csv(VAL_CSV)['wav_path']))
    if overlap:
        raise RuntimeError(f'{len(overlap)} test utterances overlap train/val')


def read_split(path, limit=None):
    if not Path(path).exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, keep_default_na=False)
    df = df[df['wav_path'].map(lambda p: Path(p).exists())].reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f'no readable wav files listed in {path}')
    if limit is not None and limit > 0:
        df = df.iloc[:limit].reset_index(drop=True)
    return df


_enc_mel = T.MelSpectrogram(sample_rate=MEL_SR, n_fft=N_FFT, win_length=WIN_LENGTH,
                            hop_length=HOP_LENGTH, n_mels=N_MELS, f_min=FMIN, f_max=FMAX,
                            power=1.0)
_voc_mel = T.MelSpectrogram(sample_rate=MEL_SR, n_fft=N_FFT, win_length=WIN_LENGTH,
                            hop_length=HOP_LENGTH, n_mels=N_MELS, f_min=FMIN, f_max=FMAX,
                            power=1.0, normalized=False, norm='slaney', mel_scale='slaney')


def read_audio(path):
    try:
        import soundfile as sf
        data, sr = sf.read(str(path), dtype='float32', always_2d=True)
        return torch.from_numpy(np.ascontiguousarray(data.T)), sr
    except ImportError:
        return torchaudio.load(str(path))


def write_audio(path, wav, sr=MEL_SR):
    wav = wav.detach().float().cpu().clamp(-1.0, 1.0)
    if wav.dim() == 2:
        wav = wav.mean(0)
    try:
        import soundfile as sf
        sf.write(str(path), wav.numpy(), sr, subtype='PCM_16')
    except ImportError:
        torchaudio.save(str(path), wav[None], sr)


def load_wav(path):
    wav, sr = read_audio(path)
    if wav.numel() == 0:
        raise RuntimeError(f'empty audio: {path}')
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != MEL_SR:
        wav = torchaudio.functional.resample(wav, sr, MEL_SR)
    return wav[:, :MAX_SAMPLES].contiguous()


def n_mel_frames(n_samples):
    return min(MAX_MEL_LEN, max(n_samples, N_FFT // 2 + 1) // HOP_LENGTH + 1)


def _min_length(wav):
    need = N_FFT // 2 + 1
    if wav.shape[-1] < need:
        wav = F.pad(wav, (0, need - wav.shape[-1]))
    return wav


def encoder_mel(wav):
    wav = _min_length(wav)
    mel = torch.log(torch.clamp(_enc_mel.to(wav.device)(wav).squeeze(0), min=1e-5))
    mel = mel[:, :MAX_MEL_LEN]
    if mel.shape[1] < MAX_MEL_LEN:
        mel = F.pad(mel, (0, MAX_MEL_LEN - mel.shape[1]), value=-11.5)
    return mel


def vocoder_mel(wav):
    wav = _min_length(wav)
    mel = torch.log(torch.clamp(_voc_mel.to(wav.device)(wav).squeeze(0), min=1e-5))
    return mel[:, :MAX_MEL_LEN]


def augment_mel(mel, generator=None):
    mel = mel.clone()
    t_len = mel.shape[-1]
    t_mask = max(1, int(0.10 * t_len))
    t0 = torch.randint(0, max(1, t_len - t_mask), (1,), generator=generator).item()
    mel[:, t0:t0 + t_mask] = -11.5
    f_mask = torch.randint(1, 16, (1,), generator=generator).item()
    f0 = torch.randint(0, N_MELS - f_mask, (1,), generator=generator).item()
    mel[f0:f0 + f_mask, :] = -11.5
    return mel


def text_to_ids(text):
    return [VOCAB[c] for c in str(text).lower() if c in VOCAB]


def ids_to_text(ids):
    return ''.join(IDX2CHAR.get(int(i), '') for i in ids)


class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ConvStem(nn.Module):
    def __init__(self, n_mels, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_mels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.downsample = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, mel):
        x = self.net[0](mel)
        x = self.net[1](x)
        x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.downsample(x)
        x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x.permute(0, 2, 1)


class CustomSpeechEncoder(nn.Module):
    def __init__(self, n_mels=N_MELS, hidden_dim=HIDDEN_DIM, num_layers=ENCODER_LAYERS,
                 nhead=ENCODER_HEADS, ffn_dim=FFN_DIM, dropout=DROPOUT):
        super().__init__()
        self.conv_stem = ConvStem(n_mels, hidden_dim)
        self.pos_enc = SinusoidalPE(hidden_dim)
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=ffn_dim,
                                           dropout=dropout, activation='gelu', batch_first=True,
                                           norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, mel):
        x = self.conv_stem(mel)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.out_norm(x)


class ReconDecoder(nn.Module):
    def __init__(self, n_mels=N_MELS, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, stride=2,
                                           padding=1, output_padding=1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Conv1d(hidden_dim, n_mels, kernel_size=1)

    def forward(self, z):
        x = z.detach().permute(0, 2, 1)
        x = self.upsample(x)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.proj(x)


class CTCHead(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, z):
        return self.proj(z)


class ModalityAdapter(nn.Module):
    def __init__(self, encoder_dim=HIDDEN_DIM, llm_dim=LLM_DIM, num_query_tokens=NUM_AUDIO_TOKENS,
                 num_qformer_layers=2, nhead=8, ffn_dim=512):
        super().__init__()
        self.conv_downsample = nn.Sequential(
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, encoder_dim))
        self.query_dropout = nn.Dropout(0.1)
        layer = nn.TransformerDecoderLayer(d_model=encoder_dim, nhead=nhead, dim_feedforward=ffn_dim,
                                           batch_first=True, activation='gelu', dropout=0.15)
        self.qformer = nn.TransformerDecoder(layer, num_layers=num_qformer_layers)
        self.norm = nn.LayerNorm(encoder_dim)
        self.linear_proj = nn.Linear(encoder_dim, llm_dim)

    def forward(self, encoder_output):
        b = encoder_output.size(0)
        x = self.conv_downsample(encoder_output.permute(0, 2, 1)).permute(0, 2, 1)
        queries = self.query_dropout(self.query_tokens.expand(b, -1, -1))
        out = self.qformer(queries, memory=x)
        return self.linear_proj(self.norm(out))


def torch_load(path, device):
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location=device, weights_only=False)


def load_encoder_checkpoint(path, device):
    ckpt = torch_load(path, device)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f'{path} is not a checkpoint dict')
    if 'encoder_state' not in ckpt:
        if 'adapter_state' in ckpt:
            raise RuntimeError(f'{path} is a Stage A checkpoint, not an encoder checkpoint')
        ckpt = {'encoder_state': ckpt}
    return ckpt


def load_frozen_encoder(device, path=ENCODER_CKPT):
    ckpt = load_encoder_checkpoint(path, device)
    encoder = CustomSpeechEncoder().to(device)
    encoder.load_state_dict(ckpt['encoder_state'])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def build_llm(device, fp16, llm_id=LLM_ID):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model
    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if (fp16 and device.type == 'cuda') else torch.float32
    llm = AutoModelForCausalLM.from_pretrained(llm_id, dtype=dtype).to(device)
    for p in llm.parameters():
        p.requires_grad = False
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=LORA_RANK, lora_alpha=LORA_ALPHA,
                     lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGETS, bias='none')
    return tokenizer, get_peft_model(llm, cfg)


def lora_state_dict(llm):
    return {k: v.detach().cpu() for k, v in llm.state_dict().items() if 'lora_' in k}


def load_lora_state(llm, state):
    if not state:
        raise RuntimeError('checkpoint has no LoRA weights')
    own = {k for k in llm.state_dict() if 'lora_' in k}
    given = {k: v for k, v in state.items() if 'lora_' in k}
    missing = own - set(given)
    if missing:
        raise RuntimeError(f'{len(missing)} LoRA tensors missing, e.g. {sorted(missing)[0]}')
    llm.load_state_dict(given, strict=False)


def load_stage_a(adapter, llm, device, path=STAGE_A_CKPT):
    ckpt = torch_load(path, device)
    if not isinstance(ckpt, dict) or 'adapter_state' not in ckpt or 'lora_state' not in ckpt:
        raise RuntimeError(f'{path} is not a Stage A checkpoint')
    adapter.load_state_dict(ckpt['adapter_state'])
    load_lora_state(llm, ckpt['lora_state'])
    return ckpt


def llm_parts(llm):
    base = llm.get_base_model()
    return base.model, base.lm_head, base.get_input_embeddings()
