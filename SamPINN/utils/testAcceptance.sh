#!/bin/bash

#perform multiple Multisolve to get an estimated mean of the acceptance ratio
#useful to test the effectiveness of new features like transfer learning

#setup the directory above (SamPINN) as working directory
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PREV_DIR=${SCRIPT_DIR%/*}

for i in {1..10}
do
    j=${i}
    bsub -cwd $PREV_DIR -oo "${j}_testAccept.out" -n 64 -W 04:00 -R "span[ptile=64]" -R "select[model==EPYC_7742]" -R "rusage[mem=500]" "python MultiSolve.py"
done

: <<'END_COMMENT'
for VARIABLE in {1...10}
do
    #run multisolve
    source lsfsolve.sh single

    sleep 30

    #wait and check if the job is finished
    while [[ `bjobs | wc -l` -lt 2 ]]
    do
        echo $VARIABLE
        echo "Running or pending"
        sleep 30

    done 
    echo "Done"

    #get the simulated acceptance ratio
    VAR="`grep Acceptance eigensolvesingle.out`"
    lastThree=${country: -3}

    echo $lastThree >> acceptances.txt


done

echo "Acceptance estimator computed"
END_COMMENT
