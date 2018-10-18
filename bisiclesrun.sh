#!/bin/bash -l

export DRIVER=$BISICLES_HOME/BISICLES/code/exec2D/driver2d.Linux.64.mpic++.gfortran.OPT.MPI.ex

export RUNDIR=`pwd`
export JOBNAME=$2
export INFILEBASE=$1

if [ -n $3 ]; then
    export JOBDIR=$3
else
    export JOBDIR=./
fi
    
mkdir -p $JOBDIR$JOBNAME

export INFILE=$INFILEBASE.$JOBNAME

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
VER=$(getgitver "/home/skachuck/work/giapy/")
echo "# GIAPY VERSION: $VER" >> $INFILE
VER=$(getgitver $RUNDIR)
echo "# GIABSICLES VERSION: $VER" >> $INFILE
VER=$(getsvnver $BISICLES_HOME/BISICLES)
echo "# BISICLES REVISION: $VER" >> $INFILE 
echo "" >> $INFILE

PYTHONPATH=./:$RUNDIR:$PYTHONPATH nohup mpirun -np 4 $DRIVER $INFILE > sout.0 &> err.0 
