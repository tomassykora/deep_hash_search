qsub -q gpu -l select=1:ncpus=1:ngpus=1:gpu_cap=cuda35:mem=2gb -l walltime=24:0:0 run.sh
