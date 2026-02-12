#!/bin/bash

# working directory (submission directory)
MYDIR="$(pwd)"
RUN_FILE="${MYDIR}/$(basename "$MYDIR").$$.pbs"

gmx="/storage/brno2/home/tedeschg/miscellaneous/gromacs_plumed2025/run/gromacs/bin/gmx_mpi"
LIBTORCH="/storage/brno2/home/tedeschg/miscellaneous/gromacs_plumed2025/plumed/libtorch"
PLUMED_LIB="/storage/brno2/home/tedeschg/miscellaneous/gromacs_plumed2025/run/plumed/lib"

# create/overwrite the PBS script file
echo "#!/bin/sh" > "$RUN_FILE"

echo "#PBS -q gpu" >> "$RUN_FILE"
echo "#PBS -l walltime=48:00:00" >> "$RUN_FILE"
echo "#PBS -l select=1:ngpus=1:ncpus=12:mem=64gb:scratch_local=100gb" >> "$RUN_FILE"

# environment variables inside the job (not only in the submit shell)
echo "export LIBTORCH=\"$LIBTORCH\"" >> "$RUN_FILE"
echo "export PLUMED_LIB=\"$PLUMED_LIB\"" >> "$RUN_FILE"
echo "export LD_LIBRARY_PATH=\"\$PLUMED_LIB:\$LIBTORCH/lib:\$LD_LIBRARY_PATH\"" >> "$RUN_FILE"

# cleanup scratch automatically on exit
echo "trap 'rm -rf \"\$SCRATCHDIR\"' TERM EXIT" >> "$RUN_FILE"

# stage-in data to scratch
echo "cp -r \"\$PBS_O_WORKDIR\"/* \"\$SCRATCHDIR\"/ || exit 1" >> "$RUN_FILE"
echo "cd \"\$SCRATCHDIR\" || exit 2" >> "$RUN_FILE"

echo "export OMP_NUM_THREADS=12" >> "$RUN_FILE"

# run the simulation
echo "$gmx grompp -f /storage/brno2/home/tedeschg/prj/cvformer/prepare/md_files/md.mdp -c /storage/brno2/home/tedeschg/prj/cvformer/05.mtd03_test-05_plain_cvformer/mtd03.gro -t /storage/brno2/home/tedeschg/prj/cvformer/05.mtd02_test-05_plain_cvformer/mtd03.cpt -p /storage/brno2/home/tedeschg/prj/cvformer/prepare/topol.top -o mtd04" >> "$RUN_FILE"
echo "$gmx mdrun -deffnm mtd04 -plumed plumed.dat" >> "$RUN_FILE"

# stage-out results back + error message if copy fails
echo "cp -r * \"\$PBS_O_WORKDIR\"/ || { trap - TERM EXIT; echo \"crashed at \$(hostname)\" >&2; exit 1; }" >> "$RUN_FILE"

# submit the job
qsub "$RUN_FILE"
