#!/bin/bash -l

#SBATCH -q regular 
#SBATCH -N 2
#SBATCH -t 24:00:00
#SBATCH -A m1041
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skachuck@umich.edu
#SBATCH --license=scratch2


export DRIVER=$BISICLES_HOME/BISICLES/code/exec2D/driver2d.Linux.64.CC.ftn.OPT.MPI.GNU.ex

export RUNDIR=`pwd`
export JOBNAME=$2
export INFILEBASE=$1
export JOBID=$SLURM_JOB_ID

if [ -n $3 ]; then
    export JOBDIR=$3
else
    export JOBDIR=./
fi
    
mkdir -p $JOBDIR$JOBNAME

export INFILE=$INFILEBASE.$JOBNAME.$JOBID

cp $INFILEBASE $JOBDIR$JOBNAME/$INFILE
cd $JOBDIR$JOBNAME

function getgitver() {
    cd $1
    local ver=`git log -n 1 | grep commit | sed s/commit\ //`
    cd $RUNDIR
    echo $ver
}

function getsvnver() {
    cd $1
    local ver=`svnversion`
    cd $RUNDIR
    echo "$ver"
}

echo "" >> $INFILE
VER=$(getgitver "/global/homes/s/skachuck/giapy/")
echo "# GIAPY VERSION: $VER" >> $INFILE
VER=$(getgitver $RUNDIR)
echo "# GIABSICLES VERSION: $VER" >> $INFILE
VER=$(getsvnver $BISICLES_HOME/BISICLES)
echo "# BISICLES REVISION: $VER" >> $INFILE 
echo "" >> $INFILE

#work out what the latest checkpoint file is (if it exists)
if test -n "$(find ./ -maxdepth 1 -name 'chk.ASE-Bedmap2.4km.l1l2.4lev.??????.2d.hdf5' -print -quit)"
    then
    LCHK=`ls -th ./chk.ASE-Bedmap2.4km.l1l2.4lev.??????.2d.hdf5 | head -n 1`
    echo "" >> $INFILE #ensure line break
    echo "amr.restart_file=$LCHK" >> $INFILE
    echo "amr.restart_set_time=false" >> $INFILE
    echo "" >> $INFILE #ensure line break
fi

PYTHONPATH=./:$RUNDIR:$RUNDIR/../:$BISICLES_HOME/BISICLES/code/libamrfile/python/AMRFile:$PYTHONPATH:$PATH 
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$BISICLES_HOME/BISICLES/code/libamrfile
srun -n 48 $DRIVER $INFILE
#srun -n 2 python -c 'from amrfile import io as amrio' 
