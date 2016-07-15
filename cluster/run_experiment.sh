# You can find the following job in the file /sge-root/examples/jobs/simple.sh.
#!/bin/sh
#
#
# (c) 2004 Sun Microsystems, Inc. Use is subject to license terms.
# This is a simple example of a SGE batch script
# request Bourne shell as shell for job
#$ -S /bin/sh
#$ -m bea
#$ -o output.txt
#$ -e error.txt
#$ -M rgarzonj@cs.upc.edu
#$ -q gpu
# Adding GPU library paths
PATH=$PATH:/usr/local/cuda/bin
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda:/usr/local/cuda/lib64
#THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python ./theano_basic_gpu_test.py
#THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python /home/usuaris/rgarzonj/github/PhD_repo/cluster/theano_basic_gpu_test.py
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python /home/usuaris/rgarzonj/github/PhD_repo/src/Pend_mc_search_theanets.py