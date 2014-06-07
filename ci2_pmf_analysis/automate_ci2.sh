#!/bin/bash

# Script for automating the collection of ci^2 and the collective
# variable location.  We assume Gard's naming convention here.

##### Variables

fbin='0'
lbin='0'
dir='/scratch/02808/dwsilver/Bulk_Glu/EVB3.1.2_largerbox_final'
prefix='bin_'
analysis='/work/02808/dwsilver/lammpsjobs/Bulk_Glu/scripts/ci2_analysis'
# Only set this to a value greater than 1 if you used a different
# output frequency for EVB and the FES file 
kline='1'

##### Main 

cd $dir

# Parse the .out and .fes files
for (( i=$fbin; $i <= $lbin; i++ )); do
  fes=$prefix$i.1.fes
  evbout=$prefix$i.1.out
  tmp1="TMP_$i"
  tmp2="TMP2_$i"
  tmp3="TMP3_$i"
  echo "Processing $fes and $evbout"
  grep -A 1 'EIGEN' $evbout | grep -v 'EIGEN' | grep -v '\-\-' > $tmp1 
  # Get the zeroth timestep from the file
  head -1 $tmp1 > $tmp2
  # Remove the first line from the file
  tail -n +2 $tmp1 > $tmp3
  # Keep the lines specified by the user
  awk "NR % $kline == 0" $tmp3 >> $tmp2
  python $analysis/get_rcec_ci2.py $fes $tmp2 
  rm $tmp1 $tmp2 $tmp3 
done
