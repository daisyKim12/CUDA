#!/bin/bash

#SBATCH -J tpc_slowdown
#SBATCH --partition=a6000
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=06:00:00

cd /nfs/home/daisy1212/X_test

./r.out 4096
./r.out 8192

./t.out 128
./t.out 256