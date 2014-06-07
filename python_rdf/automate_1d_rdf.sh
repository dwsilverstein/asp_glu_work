#!/bin/bash

# Script for running the 1D RDF script automatically

#### Variables

fwndw='9'
lwndw='36'
fprefix='bin_'
fend='.1.dump'
cecprefix='BIN_'
cecend='.1_CEC.xyz'

#### Main

# Calculate RDFs for the given US windows
for (( i=$fwndw; $i <= $lwndw; i++)); do
  f=$fprefix$i$fend
  cec=$cecprefix$i$cecend
  outpref=$cecprefix$i'_'
  echo "Making RDFs for $f"
  echo "python make_rdf_revised.py $f $cec $outpref"
  python make_rdf_revised.py $f $cec $outpref
done
