#!/bin/bash

python /home/tedeschg/prj/cvformer/testit.py \
  --trajectory traj_skip100NoH.xtc \
  --topology 2JOF-0-proteinNoH.pdb \
  --output_dir output \
  --latent_dim 2 \
  --epochs 1000 \
  --batch_size 128 \
  --patience 50 \
  --d_model 512 \
  --plumed_top npt.gro \
  --plumed_export both \
  --num_workers 6

