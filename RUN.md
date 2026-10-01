# Running Stage B and the evaluations (Colab)

Run each block in a Colab cell, from the repo root, with Drive mounted at `/content/drive`. LJSpeech-1.1 should be extracted in the working directory, and `train.csv`/`val.csv` from the encoder run should be present. `test.csv` is regenerated with the same seed (42), and the script warns if the regenerated val split differs from your `val.csv`.

```
!pip install -q speechbrain
!python protect_encoder_ckpt.py
```

Stop if this does not print `OK step=12000`.

## 1. Encoder metrics (Table I, CER, cosine metrics)

```
!python evaluate.py encoder
```

Writes `MyDrive/eval/encoder_metrics.json`.

## 2. Stage A (finish it first)

Run Cell 6 through Cell 11 of `custom_encoder_v4.py`. The output goes to `MyDrive/stage_a/`. Then run the conditioning control:

```
!python evaluate.py stage_a
```

Writes `MyDrive/eval/stage_a_control.json` with text CE for real, shuffled and zeroed audio.

## 3. Stage B training

```
!python stage_b.py train
```

Resumes from `MyDrive/stage_b/last.pt` automatically after a disconnect. Use `--fresh` to restart. Outputs go to `MyDrive/stage_b/`: `best.pt`, `last.pt`, `metrics.csv` (train and val curves) and `summary.json`.

## 4. Stage B evaluation (test split)

```
!python evaluate.py stage_b
```

Writes `MyDrive/eval/stage_b_metrics.json`, `stage_b_per_utterance.csv`, and audio samples in `stage_b_samples/`.

## 5. Voice-to-voice on any file

```
!python stage_b.py infer --wav path/to/input.wav
```

Commit the JSON and CSV files from `MyDrive/eval` and `MyDrive/stage_b` into `results/` so every number in the paper traces to a file.
