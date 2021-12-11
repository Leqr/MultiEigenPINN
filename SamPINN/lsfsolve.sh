#!/bin/bash

ncores=64
maxmem=40000
mem_per_core=$((maxmem/ncores)) 
if [ "$1" = "single" ]; then
    echo "Single core job"
    bsub -oo "eigensolvesingle.out" -n $ncores -W 04:00 -R "span[ptile=$ncores]" -R "select[model==EPYC_7742]" -R "rusage[mem=$mem_per_core]" "python MultiSolve.py"
fi

ncores=64
maxmem=40000
mem_per_core=$((maxmem/ncores))
ptile_cores=$((ncores/4))
if [ "$1" = "multi" ]; then
    echo "Multi core job"
    bsub -oo "eigensolvemulti.out" -n $ncores -W 04:00 -R "span[ptile=$ptile_cores]" -R "rusage[mem=$mem_per_core]"  "python MultiSolve.py"
fi

ncores=64
maxmem=40000
mem_per_core=$((maxmem/ncores)) 
if [ "$1" = "massive" ]; then
    echo "Massive core job"
    bsub -oo "eigensolvemassive.out" -n $ncores -W 04:00 -R "rusage[mem=$mem_per_core]" "python MultiSolve.py"
fi
