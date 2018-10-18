#!/bin/bash -l
#SBATCH -q regular
#SBATCH -N 2
#SBATCH -t 24:00:00
#SBATCH -A m1041
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skachuck@umich.edu

export DRIVER=$BISICLES_HOME/BISICLES/code/exec2D/driver2d.Linux.64.CC.ftn.OPT.MPI.GNU.ex

export INFILEBASE=./inputs.ASEBedmap2.4km.noramp.l4
export INFILE=$INFILEBASE.$SLURM_JOBID
cp $INFILEBASE $INFILE

#work out what the latest checkpoint file is (if it exists)
if test -n "$(find ./raw/ -maxdepth 1 -name 'chk.ASE-Bedmap2.4km.l1l2.4lev.??????.2d.hdf5' -print -quit)"
    then
    LCHK=`ls -th ./raw/chk.ASE-Bedmap2.4km.l1l2.4lev.??????.2d.hdf5 | head -n 1`
    echo "" >> $INFILE #ensure line break
    echo "amr.restart_file=$LCHK" >> $INFILE
    echo "amr.restart_set_time=false" >> $INFILE
    echo "" >> $INFILE #ensure line break
fi

srun -n 48 $DRIVER $INFILE
