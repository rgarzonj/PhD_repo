You can find the following job in the file /sge-root/examples/jobs/simple.sh.
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
#$ -q test
#
# print date and time
date
# Sleep for 20 seconds
sleep 20
# print date and time again
date

