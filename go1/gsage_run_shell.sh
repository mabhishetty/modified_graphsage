#!/bin/bash

#SBATCH --time=48:00:00

#SBATCH --partition=htc

#SBATCH --job-name=G_SAGE_try2

#SBATCH --ntasks-per-node=1

#SBATCH --mail-type=ALL

#SBATCH --mail-user=manoj.abhishetty@oriel.ox.ac.uk

#SBATCH --output=gsage_2.out

#SBATCH --error=gsage_2.err

#SBATCH --mem=80000

#SBATCH --constraint='cpu_gen:Skylake'

module purge

module load python/anaconda3/2019.03

source activate $DATA/myCONDAenv

export LD_LIBRARY_PATH=$DATA/myCONDAenv/lib:$LD_LIBRARY_PATH

python model.py test=data_store/reddit_graph_6_threshold_0.02_.gexf train=data_store/reddit_graph_5_threshold_0.02_.gexf test=data_store/numPy_finDiff_LABEL_s6.npy train=data_store/numPy_finDiff_LABEL_s5.npy feature_dim=12  test=data_store/numPy_finDiff_s6.npy train=data_store/numPy_finDiff_s5.npy concepts=concepts.txt concept=covid
