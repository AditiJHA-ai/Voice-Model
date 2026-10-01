import os
import shutil
import sys

import torch

SRC = '/content/drive/MyDrive/ckpt_step12000.pt'
DST = '/content/drive/MyDrive/encoder_best_step12000.pt'
REQUIRED = {'encoder_state', 'decoder_state', 'ctc_head_state'}


def main():
    if os.path.exists(DST):
        print(f'{DST} already exists, not overwriting')
        return 0
    if not os.path.exists(SRC):
        print(f'{SRC} missing')
        return 1
    try:
        ck = torch.load(SRC, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f'failed to load {SRC}: {e}')
        return 1
    keys = set(ck.keys()) if isinstance(ck, dict) else set()
    missing = REQUIRED - keys
    if missing:
        print(f'overwritten or incomplete: keys={sorted(keys)} missing={sorted(missing)}')
        return 1
    print(f'OK step={ck.get("step")} val_loss={ck.get("val_loss")}')
    shutil.copy2(SRC, DST)
    print(f'copied to {DST}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
