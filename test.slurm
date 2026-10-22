#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=test.out
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1

echo "Running on:"
hostname
echo "Done"
