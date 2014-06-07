#! /usr/bin/env python

from __future__ import print_function
import sys
import numpy as np

def main():
    """\
    Python script for correcting potential mistakes in the CEC coordinate 
    """

    if len(sys.argv) != 5:
        sys.exit('Number of inputs is incorrect')
    
    # Get the filenames
    tsfile = sys.argv[1]
    cecfile = sys.argv[2]

    # Get the stepsize
    stepsize = int(sys.argv[3])

    # Output file
    output = sys.argv[4]

    # Read the data from the .fes file.  Delete the first element, 
    # since the 0th timestep is printed twice (for some reason).  This
    # is simple with np.loadtxt, which can read columns of data.
    ts = np.loadtxt(tsfile, usecols=(1,), unpack=True, dtype=int)
    
    # Read the data from the EVB output file.  This is done by reading
    # the file into memory.
    with open(cecfile) as fl:
        f = tuple([line.rstrip() for line in fl])

    # Check that the timestep isn't one that the user didn't intend
    # to use (i.e. where the pivot state changes when you don't have
    # the output frequency set to 1 in settings.evb).
    cec = []
    for i in range(len(ts)):
        if ts[i] == 0:
            cec.append(f[i])
        else:
            if ts[i] % stepsize == 0:
                cec.append(f[i])

    # Write the CEC XYZ file
    fout = open(output, 'w')
    for i in cec:
        print(i, file=fout)
    fout.close()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
