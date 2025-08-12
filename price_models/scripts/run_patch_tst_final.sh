#!/bin/bash

DATA_ROOT="data/market_data/"


# Direct call — exactly what you'd run manually
python -u -m run \
  --model patch_tst \
  --run_name "patch_tst_final" \
  --epochs 100 \
  --lr 1e-3 \
  --batch_size 16 \
  --seed 1001 \
  --patience 15 \
  --data_root "$DATA_ROOT" \
  --lookback 30 \
  --holding_period 5 \
  --warmup_epochs 8 \
  --min_weight 0.001 \
  --components 40 \
  --utility 10.0 \
  --volatility 0.14 \
  --lc_utility 10.0 \
  --lc_components 5.5 \
  --lc_volatility 0.0 \
  --lc_min_weight 0.0 \
  --d_model 64 \
  --n_heads 4 \
  --d_ff 64 \
  --dropout 0.2 \
  --e_layers 2 \
  --factor 1 \
  --activation gelu \
  --enc_in 32 \
  --patch_len 8 \
  --stride 4 \
  --scheduler_type plateau
