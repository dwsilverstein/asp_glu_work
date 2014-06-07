#! /usr/bin/env python

from __future__ import print_function
import sys
import numpy as np

def main():
    """\
    Short Python script for grabbing the dissociation RC distance
    from a .fes file, and the largest ci^2 coefficient from an EVB
    output file.
    """

    if len(sys.argv) != 3:
        sys.exit('A filename was not supplied!')
    
    # Get the filenames
    fesfile = sys.argv[1]
    evbfile = sys.argv[2]

    # Read the data from the .fes file.  This is simple with np.loadtxt,
    # which can read columns of data.  Delete the first element, since 
    # the 0th timestep is printed twice (for some reason).
    ts, rcec = np.loadtxt(fesfile, usecols=(0,2), unpack=True)
    ts = np.delete(ts, 0)
    ts = np.array(ts, dtype=int)
    rcec = np.delete(rcec, 0)
    
    # Read the data from the EVB output file.  This is done by reading
    # the file into memory.
    with open(evbfile) as fl:
        f = tuple([line.rstrip() for line in fl])

    ci2 = []
    for ln in f:
        tmp = ln.split()
        # We use a list comprehension because the number of ci coefficients
        # depends on the number of MS-EVB states, and therefore has variable
        # length.
        ltmp = np.array([ float(tmp[i]) for i in range(len(tmp)) ])
        # Get the maximum ci coefficient, square it, and add it to
        # ci2.
        ci2.append( np.amax(ltmp) * np.amax(ltmp) )

    # Free memory
    f = 0
    tmp = 0
    ltmp = 0

    # Write data to file
    datafile = 'RCEC_CI2_' + fesfile.split('.')[0].upper()
    f = open(datafile, 'w')
    fmt = '{0:>8}  {1:>10.6f} {2:>8.5f}'
    print('# Timestep     RC        ci^2', file=f)
    for i in range(len(rcec)):
        # Try statement is used here in case the simulation is still
        # running.  If the simulation is running, you will see that
        # the length of the RC array is always larger than the ci^2
        # array and a warning is issued.
        try:
            print( fmt.format( ts[i], rcec[i], ci2[i] ), file=f )
        except IndexError:
            break
    f.close()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
