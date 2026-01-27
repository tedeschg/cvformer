#!/bin/bash

python ../testit.py \
  --trajectory traj_skip100NoH.xtc \
  --topology 2JOF-0-proteinNoH.pdb \
  --output_dir output \
  --latent_dim 2 \
  --epochs 1000 \
  --batch_size 64 \
  --patience 20 \
  --pool_type mean \
  --attn_temperature 3.0 \
  --latent_bound 2.0 \
  --jacobian_lambda 0.01 \
  --jacobian_num_probes 1