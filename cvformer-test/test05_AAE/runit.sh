#!/bin/bash
python ../../testit.py \
  --trajectory traj_skip100NoH.xtc \
  --topology 2JOF-0-proteinNoH.pdb \
  --output_dir output \
  --use_aae \
  --prior_kind gaussian \
  --latent_dim 2 \
  --epochs 400 \
  --batch_size 128 \
  --lambda_adv 0.8 \
  --lambda_adv_warmup_epochs 10 \
  --n_critic 5 \
  --critic_hidden 256 \
  --critic_depth 4 \
  --critic_lr 2e-4 \
  --lambda_gp 5.0 \
  --adv_updates_decoder \
  --patience 50 \
  --export_plumed
#ae
#python testit.py --trajectory traj.xtc --topology top.pdb --output_dir out_ae