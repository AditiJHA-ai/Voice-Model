# Draft text for the Stage B sections

Every `{...}` is a field in a results file. Fill them only from a real run. Fixed values (slots, weights, LR) come from `stage_b.py`.

## IV.C Stage B: Speech Generation Fine-Tuning

Stage B trains the language model to emit mel spectrograms. The encoder stays frozen. Training starts from the best Stage A adapter and LoRA weights, and the adapter, the LoRA parameters and a new mel prediction head are trained jointly on LJSpeech. Each training sequence is the 32 audio tokens, followed by the token embeddings of the transcript and an end-of-sequence token, followed by S = 216 learned mel query embeddings. A single linear layer maps the final hidden state of each mel query to R = 4 consecutive 80-band mel frames and 4 stop logits. The 216 queries therefore cover the 861-frame (10 s) maximum. Because the language model is causal, each mel query attends to the audio tokens, the full transcript and all earlier queries.

Targets are 80-band natural-log Slaney mel spectrograms (22,050 Hz, FFT 1,024, hop 256, 0–8 kHz), which match the input format of the frozen vocoder. The loss is

L = L1_mel + 0.2 · BCE_stop + 0.5 · CE_text,

where L1_mel is averaged over valid frames only. The stop target marks frames past the end of the utterance. CE_text is the next-token loss on the transcript and end-of-sequence token, kept so that the model still produces transcripts at inference. Training uses AdamW (betas 0.9/0.98, weight decay 0.01) with learning rate 1e-4 for the adapter and LoRA and 1e-3 for the mel head. The schedule is 500 warm-up steps followed by cosine decay to 10% of the peak. Batch size is 4 with gradient accumulation 8 (effective 32), with FP16 and gradient checkpointing. Validation runs every 1,000 steps with early-stopping patience 5, for at most 20,000 steps.

At inference the model first greedily decodes a transcript from the audio tokens until it emits end-of-sequence. It then appends the 216 mel queries and predicts the mel spectrogram, which is cut at the first frame whose stop probability exceeds 0.5. The pretrained LJSpeech HiFi-GAN from SpeechBrain (`speechbrain/tts-hifigan-ljspeech`), kept frozen, converts it to a waveform. The output is therefore a resynthesis of the input utterance in the LJSpeech voice.

Section III.D should say "linear mel prediction head over learned mel query positions".

## V.D Stage B and Vocoder Results

Stage B reaches its best validation loss of {stage_b/summary.json: best_val_total} at step {best_step} (mel L1 {best_val.mel_l1}, stop BCE {best_val.stop_bce}, text CE {best_val.text_ce}). Table III reports results on the {eval/stage_b_metrics.json: utterances}-utterance test split. Intelligibility is the WER of the {asr_judge} transcription. MCD is computed with DTW on 24 mel-cepstral coefficients, using the definition in `stage_b_metrics.json`. This MCD is not directly comparable to SPTK-based MCD values in other papers.

| System | Judge WER | Judge CER | MCD-DTW (dB) |
|---|---|---|---|
| Ground-truth recording | {judge_wer.ground_truth} | {judge_cer.ground_truth} | – |
| Copy synthesis (GT mel → HiFi-GAN) | {judge_wer.copy_synthesis} | {judge_cer.copy_synthesis} | {mcd_dtw_db.copy_synthesis.mean} |
| Stage B, reference transcript | {judge_wer.stage_b_gt_text} | {judge_cer.stage_b_gt_text} | {mcd_dtw_db.stage_b_gt_text.mean} |
| Stage B, full voice-to-voice | {judge_wer.stage_b_v2v} | {judge_cer.stage_b_v2v} | {mcd_dtw_db.stage_b_v2v.mean} |

The intermediate LLM transcript has WER {llm_transcript.wer} and CER {llm_transcript.cer}.

Audio-conditioning control (teacher-forced, reference transcript):

| Audio tokens | Mel L1 | Abs. length error (frames) |
|---|---|---|
| Matching utterance | {teacher_forced.real.mel_l1.mean} | {teacher_forced.real.abs_length_error_frames.mean} |
| Different utterance | {teacher_forced.shuffled.mel_l1.mean} | {teacher_forced.shuffled.abs_length_error_frames.mean} |
| Zeros | {teacher_forced.zero.mel_l1.mean} | {teacher_forced.zero.abs_length_error_frames.mean} |

Interpretation: claim that the audio conditions the output only if the matching row is clearly better than both controls. The same rule applies to Stage A, using the `text_ce` values in `stage_a_control.json`.
