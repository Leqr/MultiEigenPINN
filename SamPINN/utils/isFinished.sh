if [[ `bjobs | wc -l` -ge 2 ]] ; then
    echo False
else
    echo True
fi
