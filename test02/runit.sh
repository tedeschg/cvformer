#!/bin/bash

python ../testit.py \
  --trajectory /home/tedeschg/prj/cvformer/test11/traj_skip100NoH.xtc \
  --topology /home/tedeschg/prj/cvformer/test11/2JOF-0-proteinNoH.pdb \
  --output_dir output \
  --latent_dim 2 \
  --epochs 1000 \
  --batch_size 64 \
  --patience 20
