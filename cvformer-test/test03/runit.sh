#!/bin/bash

python /home/tedeschg/prj/cvformer/testit.py \
  --trajectory traj_skip100NoH.xtc \
  --topology 2JOF-0-proteinNoH.pdb \
  --output_dir output \
  --latent_dim 2 \
  --epochs 1000 \
  --batch_size 64 \
  --patience 20
