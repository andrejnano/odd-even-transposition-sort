#!/bin/bash

if [ $# -lt 1 ];then
  numbers=10;
else
  numbers=$1;
fi;

mpic++ -o oets ots.cpp
dd if=/dev/random bs=1 count=$numbers of=numbers >& /dev/null
PMIX_MCA_gds=^ds12 mpirun -mca shmem posix --oversubscribe -np $numbers oets
rm -f oets numbers
