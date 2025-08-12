#!/bin/bash

DATA_ROOT="data/market_data/"

python -u -m run \
  --model bidirectional_lstm \
  --run_name "bidirectional_lstm_final" \
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
  --stem_dims 16 \
  --lstm_hidden_dim 64 \
  --lstm_layers 2 \
  --head_dims 32 1 \
  --scheduler_type plateau
