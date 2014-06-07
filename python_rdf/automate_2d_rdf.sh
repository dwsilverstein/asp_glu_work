#!/bin/bash
#SBATCH -p normal 
#SBATCH -n 16
#SBATCH --job-name=2D_RDF
#SBATCH --output=log.2D_RDF.%j.out
#SBATCH --time=02:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dwsilverstein@gmail.com

# Script for creating files for 2D RDFs.  This should be run after
# generating the CEC XYZ files with make_cec_xyzfile.sh

##### Variables

# First
#fbin='0'        # First US window
#lbin='7'        # Last US window
# Second
#fbin='8'         # First US window
#lbin='15'        # Last US window
# Third
#fbin='16'        # First US window
#lbin='23'        # Last US window
# Fourth
#fbin='24'        # First US window
#lbin='31'        # Last US window
# Fifth
fbin='32'        # First US window
lbin='36'        # Last US window
fprefix='bin_'   # EVB output file prefix
xyzprefix='BIN_' # Prefix for XYZ file
# Directory containing our data
dir='/scratch/02808/dwsilver/Bulk_Glu/EVB3.1.2_largerbox_final'
# Location of the executables (make_2d_rdf.py)
exe='/work/02808/dwsilver/lammpsjobs/Bulk_Glu/scripts/python_rdf'
# First and last timestep to collect for calculating RDFs (this is
# the LAMMPS timestep divided by the output frequency)
fts='1000'
lts='19000'

##### Main

# Go to the directory with the data
cd $dir

ndx='0'
for (( i=$fbin; $i <= $lbin; i++ )); do
  # Dump file, CEC XYZ file, and prefix for naming the RDF files
  dump=$fprefix$i.1.dump
  cecxyz=$xyzprefix$i.1_CEC.xyz
  rdfpref='BIN_'$i'_2D_'

  # Run the 2D RDF script
  echo "Running US window $i"
  taskset -c $ndx python $exe/make_2d_rdf.py $dump $cecxyz $rdfpref $fts $lts &
  
  # Counter for running jobs on a compute node
  ndx=$((ndx+1))
  if [[ "$((ndx%4))" -eq '0' ]]; then
    echo "Waiting at index $ndx..."
    wait
    ndx=0
    echo
  fi
done
# Make sure everything gets to finish
if [[ "$ndx" -ne '0' ]]; then
  echo "Waiting at index $ndx..."
  wait
  echo
fi
