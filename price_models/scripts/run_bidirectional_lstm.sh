#!/bin/bash

DATA_ROOT="data/market_data/"

stem_dims=(
  "16"
  "64"
  "64 16"
)
head_dims_list=(
  "1"
  "32 1"
)

i=0
for stem_dim in "${stem_dims[@]}"; do
  for lstm_hidd_dim in 64 128; do
    for lstm_layers in 2; do
      for head_dim in "${head_dims_list[@]}"; do
          run_name="lstm_stem_dims_${stem_dim}_lstm_hidd_dim_${lstm_hidd_dim}_lstm_layers_${lstm_layers}_head_dims_${head_dim}"
          echo -e "\n[$i] Running: $run_name"

          python -u -m run \
            --model bidirectional_lstm \
            --run_name "$run_name" \
            --epochs 100 \
            --lr 1e-3 \
            --batch_size 16 \
            --seed 1001 \
            --patience 5 \
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
            --stem_dims $stem_dim \
            --lstm_hidden_dim $lstm_hidd_dim \
            --lstm_layers $lstm_layers \
            --head_dims $head_dim \
            --scheduler_type plateau

          i=$((i + 1))
      done
    done
  done
done
