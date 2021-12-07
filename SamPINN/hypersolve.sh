#!/bin/bash
if [ "$1" = "single" ]; then
    echo "Single core job"
    bsub -oo "eigensolvesingle.out" -n 48 -W 06:00  -R "span[ptile=48]" -M 8G "python MultiSolve.py"
fi

if [ "$1" = "multi" ]; then
    echo "Multi core job"
    bsub -oo "eigensolvemulti.out" -n 48 -W 06:00 -R "span[ptile=12]" -M 8G "python MultiSolve.py"
fi

#-R "select[model==EPYC_7742]"
