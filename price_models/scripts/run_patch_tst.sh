#!/bin/bash

DATA_ROOT="data/market_data/"

i=0
for d_model in 64 128; do
  for n_heads in 2 4 8; do
    for d_ff in 64 128 256; do
      for dropout in 0.1 0.2; do
        for e_layers in 2 3; do
          run_name="patchtst_d_model_${d_model}_n_heads_${n_heads}_d_ff_${d_ff}_dropout_${dropout}_e_layers_${e_layers}"
          echo -e "\n[$i] Running: $run_name"

          # Direct call — exactly what you'd run manually
          python -u -m run \
            --model patch_tst \
            --run_name "$run_name" \
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
            --components 50 \
            --utility 10.0 \
            --volatility 0.14 \
            --lc_utility 10.0 \
            --lc_components 5.0 \
            --lc_volatility 0.0 \
            --lc_min_weight 0.0 \
            --d_model $d_model \
            --n_heads $n_heads \
            --d_ff $d_ff \
            --dropout $dropout \
            --e_layers $e_layers \
            --factor 1 \
            --activation gelu \
            --enc_in 32 \
            --patch_len 8 \
            --stride 4 \
            --scheduler_type plateau

          i=$((i + 1))
        done
      done
    done
  done
done