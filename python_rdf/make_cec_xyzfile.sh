#!/bin/bash

# Script for making a CEC file with its XYZ coordinates

##### Variables

# Timesteps to keep.  This should in general be set to
# match the frequency of the dump file
kline='100'        
fbin='0'            # First US window
lbin='36'           # Last US window
fprefix='bin_'      # EVB output file prefix
xyzprefix='BIN_'    # Prefix for XYZ file
xyzend='.1_CEC.xyz' # Ending for XYZ file
dir='/scratch/02808/dwsilver/Bulk_Glu/EVB3.1.2_largerbox_final'
# Location of the get_timestep_rcec.py script
EXE='/work/02808/dwsilver/lammpsjobs/Bulk_Glu/scripts/python_rdf'

##### Main

cd $dir

# Make CEC XYZ files for each US window
for (( i=$fbin; $i <= $lbin; i++ )); do
  f=$fprefix$i.1.out
  # Temporary files for CEC coordinates
  tmp=TMP$i.xyz
  # Temporary file for timesteps
  ttmp=TTMP$i.xyz
  fxyz=$xyzprefix$i.1_CEC.xyz
  echo "Processing $f into $fxyz" 
  # Grab the CEC coordinate
  grep -A 1 'CEC_C' $f | grep -v 'CEC' | grep -v '\-\-' > $tmp
  # Grab the timestep
  grep 'TIMESTEP' $f > $ttmp
  # Keep the lines specified by the user
  python $EXE/get_timestep_rcec.py $ttmp $tmp $kline $xyzprefix$i$xyzend
  # Remove the extra files
  rm $tmp
  rm $ttmp
done
