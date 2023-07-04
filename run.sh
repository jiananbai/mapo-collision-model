#!/bin/bash
#
#SBATCH -J MAPO
#SBATCH -t 23:59:59
#SBATCH -n 32
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jianan.bai@liu.se
#
module load Python/3.8.3-anaconda-2020.07-extras-nsc1
source activate mapo
for i in $(seq 8)
do
    srun --exclusive --nodes=1 --ntasks=1 python -u main.py -v -s --feedback --policy mapo > outputs/output-mapo-$i.txt &
sleep 10
done
for i in $(seq 8)
do
    srun --exclusive --nodes=1 --ntasks=1 python -u main.py -v -s --feedback --policy qmix > outputs/output-qmix-$i.txt &
sleep 10
done
for i in $(seq 8)
do
    srun --exclusive --nodes=1 --ntasks=1 python -u main.py -v -s --feedback --pre_alloc --policy mapo > outputs/output-mapo-$i.txt &
sleep 10
done
for i in $(seq 8)
do
    srun --exclusive --nodes=1 --ntasks=1 python -u main.py -v -s --feedback --pre_alloc --policy qmix > outputs/output-qmix-$i.txt &
sleep 10
done
wait

# Script ends here
