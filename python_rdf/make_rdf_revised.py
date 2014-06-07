#! /usr/bin/env python

from __future__ import print_function, division
import sys
from numpy import array, histogram, linspace, pi, sqrt, ceil, power
from datetime import datetime

def main():
    """\
    Script for producing RDFs.  The steps are:

    (1) Collect the data from the LAMMPS trajectory (dump) file.

    (2) Collect the data from the CEC coordinate file.

    (3) Calculate distances between the atom centers and bin the data

    (4) Normalize the data based on an ideal system.

    (5) Output RDFs. 

    NOTE!!!  The OW-OW, OW-HW, and HW-HW RDF calculation is very slow.
    There isn't much we can do about this, unfortunately, unless we 
    reduce the number of timesteps sampled, or write the code more cleverly
    by embedding C/C++ code here with Scipy's weave method.
    """

    # Filename prefix for output (not required)
    if len(sys.argv) == 4:
        fprefix = sys.argv[3]
    else:
        fprefix = ''

    # Check that the number inputs is correct.
    if len(sys.argv) < 3:
        error1 = 'Expected 2 files but was given ' + str(len(sys.argv)-1)
        nl = '\n'
        tb = '='*len(error1)
        sys.exit(nl + tb + nl + error1 + nl + tb + nl)

    # Check that the first file is a trajectory file.
    test = open(sys.argv[1]).readline().rstrip()
    if test != 'ITEM: TIMESTEP':
        error2 = 'The first file must be a LAMMPS trajectory (dump) file'
        nl = '\n'
        tb = '='*len(error2)
        sys.exit(nl + tb + nl + error2 + nl + tb + nl)

    # Turn on/off the water-water RDFs
    do_water = False
    #do_water = True

    # This part is specific to Glu and Asp at the moment
    # ----------------------------------------
    # GLU-P or ASP-P with H2O
    # 3 = methylene C next to alpha C
    # 4 = H on alpha carbon (methine group)
    # 5 = methylene C next to carboxylic acid
    # 6 = methyl or methylene H
    # 8 = C carbonyl of amide group
    # 9 = O in carbonyl of amide group
    # 10 = N of amide group
    # 11 = H on polar group (N-H or O-H)
    # 14 = O from -OH group of carboxylic acid
    # 15 = C in methyl group
    # 36 = C in carboxylic acid
    # 37 = carbonyl O of carboxylic acid
    # 38 = Ow
    # 39 = Hw
    # ----------------------------------------
    # GLU-D or ASP-D with H3O+
    # 3 = methylene C next to alpha C
    # 4 = H on alpha carbon (methine group)
    # 5 = methylene C next to carboxylic acid
    # 6 = methyl or methylene H
    # 8 = C carbonyl in amide
    # 9 = O in carbonyl of amide group
    # 10 = N of amide group
    # 11 = H on polar group (N-H or O-H)
    # 15 = C in methyl group
    # 20 = C in COO-
    # 28 = O in COO-
    # 38 = Ow
    # 39 = Hw
    # 40 = O (H3O)
    # 41 = H (H3O)
    # ----------------------------------------

    # The fact that we use the same atom type for all atoms in Glu-P
    # and Glu-D can cause problems here.  Be careful if you want an
    # RDF in the region where the bonding topology is changing,

    # List of important atoms
    # ****** Protonated Amino Acid ******
    # Atoms from COOH 
    prot = [ '36', '14', '37' ]
    prot_com = [ '36', '14', '37' ] # Center of mass atoms
    # ****** Deprotonated Amino Acid ******
    # Atoms from COO-
    deprot = [ '20', '28' ]
    deprot_com = [ '20', '28' ] # Center of mass atoms
    # ****** Water ******
    water = [ '38', '39' ]
    # ****** Hydronium ******
    hydronium = [ '40', '41' ]
    # ****** CEC ******
    # This index is simply chosen to not correpsond to any atom indices
    # present in the LAMMPS trajectory file (the value is arbitrary).
    # I choose a negative index since this cannot be chosen in LAMMPS 
    cec = [ '-1' ]
    # ****** Backbone ******
    # WARNING: I haven't come up with a way to distinguish backbone
    # atoms in the protonated and deprotonated amino acids.  Not sure
    # if this is an issue. 
    # All backbone atoms
    #backbone = [ '3', '4', '5', '6', '8', '9', '10',
    #         '11', '15' ]
    # Amide oxygen
    backbone = [ '9' ]

    # Masses for determining the center of mass based on the carboxylate
    # group.
    mass = array([ 12.011, 15.999, 15.999 ])
    totalmass = sum(mass)
    mass = mass / totalmass

    # Atoms to search for
    atomsearch = []
    for i in prot:
        atomsearch.append(i)
    for i in deprot:
        atomsearch.append(i)
    for i in water:
        atomsearch.append(i)
    for i in hydronium:
        atomsearch.append(i)
    for i in backbone:
        atomsearch.append(i)

    # Atom number to atom name correspondence (arbitrary)
    # Use this for naming files.
    table = { '9'  : 'OAM',
              '14' : 'OP',
              '36' : 'CP',
              '37' : 'OC',
              '20' : 'CD',
              '28' : 'OA',
              '38' : 'OW',
              '39' : 'HW',
              '40' : 'OH',
              '41' : 'HO',
              '-1' : 'CC',   # CEC
              '-2' : 'PCOM', # Protonated amino acid center of mass
              '-3' : 'DCOM', # Deprotonated amino acid center of mass
            }

    # Get the filenames
    trajfile = sys.argv[1]
    cecfile = sys.argv[2]

    # Start the timer (for total time)
    startTime = datetime.now() 

    #####################################################
    # Collect data from the LAMMPS trajectory (dump) file
    #####################################################

    # Timer for LAMMPS collection
    lammpsTime = datetime.now()

    # Read the trajectory file into memory.  Store as a tuple to 
    # prevent the user from accidentally changing the values.
    with open(trajfile) as fl:
        f = tuple([line.rstrip() for line in fl])

    # Get the box volume
    x = f[5].split()               # 6th line has x-dimensions
    lx = float(x[1]) - float(x[0])
    y = f[6].split()               # 7th line has y-dimensions
    ly = float(y[1]) - float(y[0])
    z = f[7].split()               # 8th line has z-dimensions
    lz = float(z[1]) - float(z[0])
    # Total volume of the box
    vtotal = lx * ly * lz

    # Side length for minimum image convention
    side = ( lx + ly + lz ) / 3.0
    box_half = side / 2.0

    # Determine what the user put in their dump file
    headings = ( "ITEM: ATOMS id type q x y z",
                 "ITEM: ATOMS id mol type q x y z",
               )
    # Determine how the data should be collected
    collect = -1
    for i in range(len(headings)):
        if f[8] == headings[i]:
            collect = i

    # Get the number of atoms (this is always in the same place in the
    # file)
    natoms = int(f[3])
    # Rather than reading the file, use the fact that the data is 
    # distributed in regular intervals throughout the file.
    noncoordlines = 9 # Ignores the lines defining timesteps and the box
    filelength = len(f)   
    ntimesteps = int( filelength / ( natoms + noncoordlines) )
    #### DEBUG
    #ntimesteps = 10
    #### DEBUG
    # coordblocks contains the start and end of each coordinate block
    coordblocks = [ [ i * (natoms + noncoordlines) + noncoordlines, 
                      (i + 1) * (natoms + noncoordlines) ] 
                    for i in range(ntimesteps) ] 

    # Store the coordinates in a dictionary of lists
    coords = {}
    # Store the number of each atom type in a dictionary
    nattype = {}
    for i in atomsearch:
        coords[i] = [ [] for x in range(ntimesteps) ]
        nattype[table[i]] = 0
    # Gather coordinates from the blocks.  Although we don't explicitly
    # store the timestep, each list corresponds to a timestep
    timestep = 0
    for l in coordblocks:
        s = l[0]
        e = l[1]
        for i in range(s,e):
            # The atom index depends on how the dump file was output
            if collect == 0: 
                at = f[i].split()[1]
            elif collect == 1: 
                at = f[i].split()[2]
            # Skip the atom if it isn't in one of the lists above 
            # (prot, deprot, water, or hydronium).  This avoids
            # wasting time and memory storing data we don't care about.
            if at in coords.keys():
                # x-, y-, and z- coordinates
                if collect == 0:
                    tmp = [ float(f[i].split()[3]),
                            float(f[i].split()[4]),
                            float(f[i].split()[5]) ]
                elif collect == 1:
                    tmp = [ float(f[i].split()[4]),
                            float(f[i].split()[5]),
                            float(f[i].split()[6]) ]
                coords[at][timestep].append(array(tmp))
                # Get the number of atoms for each type
                nattype[table[at]] += 1 
        timestep += 1

    # Find timesteps where the (de)protonated amino acid is not
    # present.  This is needed in US windows where the bonding 
    # topology changes at different timesteps.  The code gets 
    # confused otherwise. 
    empty_prot   = [i for i,x in enumerate(coords['36']) if x == []]
    empty_deprot = [i for i,x in enumerate(coords['20']) if x == []]

    # Correct the number of each atomtype
    for key in coords.keys():
        if nattype[table[key]] != 0:
            # Protonated amino acid
            if key in prot:
                nattype[table[key]] /= ntimesteps - len(empty_prot)
            # Deprotonated amino acid
            elif key in deprot:
                nattype[table[key]] /= ntimesteps - len(empty_deprot)
            # Hydronium
            elif key in hydronium:
                nattype[table[key]] /= ntimesteps - len(empty_deprot)
            # Water
            elif key in water:
                nattype[table[key]] /= ntimesteps 
            # Backbone atoms
            elif key in backbone:
                nattype[table[key]] /= ntimesteps 
            # Convert to integers
            nattype[table[key]] = int(nattype[table[key]])

    # Check that the number of empty amino acid coordinates sums to
    # the number of timesteps.  Die here if not.
    if ( len(empty_prot) + len(empty_deprot) != ntimesteps):
        sys.exit('Coordinate collection error!') 

    # Free up memory from storing the trajectory file (we don't need 
    # the file anymore).
    f = 0
    fl.close()

    # Print out the LAMMPS data collection timing
    timing( lammpsTime, datetime.now(), 'LAMMPS collection time:' )

    ###########################################
    # Collect data from the CEC coordinate file 
    ###########################################

    # Timer for CEC collection
    cecTime = datetime.now()

    # Add the CEC label
    atomsearch.append(cec[0])
    nattype['CC'] = 1

    # Read the CEC coordinate file into memory.  Store as a tuple to
    # prevent the user from accidentally changing any values.
    with open(cecfile) as fl:
        f = tuple([line.rstrip() for line in fl])

    # Store the CEC coordinates in the coords dictionary
    ###########################################################
    # Old way of doing this
    #coords[cec[0]] = [ [] for x in range(ntimesteps) ]
    #for l in range(ntimesteps):
    #    # 0, 1, 2 are the x-, y-, and z- coordinates
    #    tmp = [ float(f[l].split()[0]),
    #            float(f[l].split()[1]),
    #            float(f[l].split()[2]) ]
    #    coords[cec[0]][l].append(array(tmp))
    ###########################################################
    # Faster method of storing the CEC coordinates
    coords[cec[0]] = [ [ array( [ float(f[l].split()[0]),
                                  float(f[l].split()[1]),
                                  float(f[l].split()[2]) ] ) ] 
                       for l in range(ntimesteps) ]
    ###########################################################

    # Free up memory from storing the CEC coordinate file 
    f = 0
    fl.close()

    # Print out the CEC coordinate collection timing
    timing( cecTime, datetime.now(), 'CEC collection time:' )

    ##############################
    # Determine the center of mass
    ##############################

    # Timer for COM calculation
    comTime = datetime.now()

    # Remove any empty dictionary keys from coords and nattype.  Also 
    # remove atoms from the atomsearch.
    for i in coords.keys():
        if nattype[table[i]] == 0:
            coords.pop(i, None)
            atomsearch.remove(i)
            nattype.pop(table[i], None)

    # Determine the center of mass coordinate
    # ****WARNING: The COM calculation is hardcoded!
    # Error check here to avoid wasting time
    com_error = True
    if ( '36' in atomsearch and
         '14' in atomsearch and
         '37' in atomsearch ):
        com_error = False
        # Add COM to atomsearch
        atomsearch.append('-2')
        # Initialize the number of COM atoms (always 1 in every timestep)
        nattype['PCOM'] = 1
        # List of lists for the COM
        #coords['-2'] = [ [] for x in range(len(coords['36'])) ]
        coords['-2'] = [ [] for x in range(ntimesteps) ]
        # Get the COM from COOH atoms (36, 14, 37).  Recall that
        # we divided by the total mass above.  For the protonated
        # amino acids, there is only one of each type
        for i in range(ntimesteps):
            # Check that the carboxylic acid exists in the current
            # timestep.
            if i not in empty_prot:
                # Minimum image distance for C
                cx = coords['36'][i][0][0] - coords['-1'][i][0][0] 
                cy = coords['36'][i][0][1] - coords['-1'][i][0][1] 
                cz = coords['36'][i][0][2] - coords['-1'][i][0][2] 
                if cx >= box_half: coords['36'][i][0][0] -= side
                if cx <= -box_half: coords['36'][i][0][0] += side
                if cy >= box_half: coords['36'][i][0][1] -= side
                if cy <= -box_half: coords['36'][i][0][1] += side
                if cz >= box_half: coords['36'][i][0][2] -= side
                if cz <= -box_half: coords['36'][i][0][2] += side
                # Minimum image distance for O1 
                cx = coords['14'][i][0][0] - coords['-1'][i][0][0]
                cy = coords['14'][i][0][1] - coords['-1'][i][0][1]
                cz = coords['14'][i][0][2] - coords['-1'][i][0][2]
                if cx >= box_half: coords['14'][i][0][0] -= side
                if cx <= -box_half: coords['14'][i][0][0] += side
                if cy >= box_half: coords['14'][i][0][1] -= side
                if cy <= -box_half: coords['14'][i][0][1] += side
                if cz >= box_half: coords['14'][i][0][2] -= side
                if cz <= -box_half: coords['14'][i][0][2] += side
                # Minimum image distance for O2 
                cx = coords['37'][i][0][0] - coords['-1'][i][0][0]
                cy = coords['37'][i][0][1] - coords['-1'][i][0][1]
                cz = coords['37'][i][0][2] - coords['-1'][i][0][2]
                if cx >= box_half: coords['37'][i][0][0] -= side
                if cx <= -box_half: coords['37'][i][0][0] += side
                if cy >= box_half: coords['37'][i][0][1] -= side
                if cy <= -box_half: coords['37'][i][0][1] += side
                if cz >= box_half: coords['37'][i][0][2] -= side
                if cz <= -box_half: coords['37'][i][0][2] += side
                # Construct the COM array
                coords['-2'][i].append( mass[0] * coords['36'][i][0]
                                      + mass[1] * coords['14'][i][0]
                                      + mass[2] * coords['37'][i][0] )
        # Remove the COOH atoms from coords, nattype, and atomsearch
        for i in [ '36', '14', '37' ]:
            coords.pop(i, None)
            atomsearch.remove(i)
            nattype.pop(table[i], None)
    if ( '20' in atomsearch and
         '28' in atomsearch ):
        com_error = False
        # Add COM to atomsearch
        atomsearch.append('-3')
        # Initialize the number of COM atoms (always 1)
        nattype['DCOM'] = 1
        # Initialize the PCOM coordinates
        coords['-3'] = [ [] for x in range(ntimesteps) ]
        # Get the COM from COO- atoms (20, 28).  Recall that
        # we divided by the total mass above.  For the deprotonated
        # amino acids, there are two carboxylate oxygens. 
        #for i in range(len(coords['20'])):
        for i in range(ntimesteps):
            # Check that the carboxylate exists in the current timestep
            if i not in empty_deprot:
                # Minimum image distance for C
                cx = coords['20'][i][0][0] - coords['-1'][i][0][0]
                cy = coords['20'][i][0][1] - coords['-1'][i][0][1]
                cz = coords['20'][i][0][2] - coords['-1'][i][0][2]
                if cx >= box_half: coords['20'][i][0][0] -= side
                if cx <= -box_half: coords['20'][i][0][0] += side
                if cy >= box_half: coords['20'][i][0][1] -= side
                if cy <= -box_half: coords['20'][i][0][1] += side
                if cz >= box_half: coords['20'][i][0][2] -= side
                if cz <= -box_half: coords['20'][i][0][2] += side
                # Minimum image distance for O1 
                cx = coords['28'][i][0][0] - coords['-1'][i][0][0]
                cy = coords['28'][i][0][1] - coords['-1'][i][0][1]
                cz = coords['28'][i][0][2] - coords['-1'][i][0][2]
                if cx >= box_half: coords['28'][i][0][0] -= side
                if cx <= -box_half: coords['28'][i][0][0] += side
                if cy >= box_half: coords['28'][i][0][1] -= side
                if cy <= -box_half: coords['28'][i][0][1] += side
                if cz >= box_half: coords['28'][i][0][2] -= side
                if cz <= -box_half: coords['28'][i][0][2] += side
                # Minimum image distance for O2 
                cx = coords['28'][i][1][0] - coords['-1'][i][0][0]
                cy = coords['28'][i][1][1] - coords['-1'][i][0][1]
                cz = coords['28'][i][1][2] - coords['-1'][i][0][2]
                if cx >= box_half: coords['28'][i][1][0] -= side
                if cx <= -box_half: coords['28'][i][1][0] += side
                if cy >= box_half: coords['28'][i][1][1] -= side
                if cy <= -box_half: coords['28'][i][1][1] += side
                if cz >= box_half: coords['28'][i][1][2] -= side
                if cz <= -box_half: coords['28'][i][1][2] += side
                # Construct the COM array
                coords['-3'][i].append( mass[0] * coords['20'][i][0]
                                      + mass[1] * coords['28'][i][0]
                                      + mass[2] * coords['28'][i][1] )
        # Remove the COO- atoms from coords, nattype, and atomsearch
        for i in [ '20', '28' ]:
            coords.pop(i, None)
            atomsearch.remove(i)
            nattype.pop(table[i], None)
    # Exit if the center of mass atoms weren't located.
    if (com_error):
        sys.exit('Could not locate COM atoms')

    # Print out the CEC coordinate collection timing
    timing( comTime, datetime.now(), 'COM calculation time:' )

    ################################################################
    # Calculate distances between atom pairs and generate histograms
    ################################################################

    # Timer for distance calculation 
    distTime = datetime.now()

    # Create pair lists for RDFs
    plist = []
    for i in atomsearch:
        # Make RDFs with water atoms
        if i not in water:
            for j in water:
                plist.append([i,j])
        # For deprotonated amino acid, make RDFs with hydronium also
        if i in deprot:
            for j in hydronium:
                plist.append([i,j])
                # For backbone atoms, make RDFs with hydronium
                for k in backbone:
                    plist.append([k,j])
        # For making RDFs between the CEC and atoms in the amino acid
        if i in cec:
            for j in prot:
                if j in atomsearch:
                    plist.append([i,j])
            for j in deprot:
                if j in atomsearch:
                    plist.append([i,j])
            for j in backbone:
                if j in atomsearch:
                    plist.append([i,j])
        # COM with CEC
        if i == '-2':
            plist.append([i,'-1'])
        if i == '-3':
            plist.append([i,'-1'])
    # Add pair lists for water atoms explicitly (this prevents double
    # counting of the OW-HW RDF without requiring a loop).
    if do_water:
        plist.append([water[0],water[0]])
        plist.append([water[0],water[1]])
        plist.append([water[1],water[1]])

    # Make the the histogram and g(r) normalization factor dictionaries 
    # (for later).  Also construct a dictionary containing the number
    # of atom pairs
    hist = {}
    norm = {}
    for pair in plist:
        # Here we use the table specified above to rename atoms with
        # letters (e.g. 'OW') instead of atom indices (e.g. '38')
        p = table[pair[0]] + '-' + table[pair[1]]
        hist[p] = 0
        norm[p] = []

    # Minimum and maximum distance for RDF, number of bins
    mindist = 0.0
    maxdist = 20.0
    nbins = 401
    #maxbin = nbins - 1

    # Bins for histogramming the data
    bin_r = linspace(mindist, maxdist, nbins) 
    #dr = bin_r[1] - bin_r[0]

    # Determine the distance between each point and bin the data
    for pair in plist:
        at1 = pair[0]
        at2 = pair[1]
        p = table[at1] + '-' + table[at2]
        tmp = datetime.now()
        h = 0
        # Check if this is the first time the histogram is being added to
        first_add = True
        for ts in range(ntimesteps):
            # Check if we're dealing with one of the COM coordinates.
            # If so, make sure the coordinate exists during this timestep.
            if ( (ts in empty_prot) and (at1 == '-2') ): continue
            if ( (ts in empty_deprot) and (at1 == '-3') ): continue
            if ( (ts in empty_deprot) and (at1 in hydronium) ): continue
            if ( (ts in empty_deprot) and (at2 in hydronium) ): continue
            ###########################################################
            # Differences in the x-, y-, and z- directions
            mind = []
            for i in range(len(coords[at1][ts])):
                for j in range(len(coords[at2][ts])):
                    # Get differences in x-, y-, and z-directions
                    x = coords[at1][ts][i][0] - coords[at2][ts][j][0]
                    y = coords[at1][ts][i][1] - coords[at2][ts][j][1]
                    z = coords[at1][ts][i][2] - coords[at2][ts][j][2]
                    # Minimum image distances
                    #### DEBUG
                    #if (x > 0) and (x > box_half):
                    #    x -= side
                    #elif (x < 0) and (abs(x) > box_half): 
                    #    x += side
                    #if (y > 0) and (y > box_half):
                    #    y -= side
                    #elif (y < 0) and (abs(y) > box_half): 
                    #    y += side
                    #if (z > 0) and (z > box_half):
                    #    z -= side
                    #elif (z < 0) and (abs(z) > box_half): 
                    #    z += side
                    #### DEBUG
                    if x > box_half: x -= side
                    if x < -box_half: x += side
                    if y > box_half: y -= side
                    if y < -box_half: y += side
                    if z > box_half: z -= side
                    if z < -box_half: z += side
                    mind.append( x*x + y*y + z*z )
            ###########################################################
            # Get the distance by taking the sqrt of mind.  Do all 
            # square roots together with NumPy, which is faster than
            # doing them as part of the double loop
            mind = sqrt(mind)
            ###########################################################
            # Use NumPy's histogram to bin the data
            h, bin_r = histogram( mind, bins=bin_r )
            # Below are two methods for binning the data.  One does
            # the binning exactly how Numpy does it.  The other does 
            # it a different way that I found in someone else's code.
            #### DEBUG
            # This is identical to NumPy's histogram
            #h = array([0]*( maxbin ))
            #for tttt in range(1,len(bin_r)): 
            #    if tttt < len(bin_r):
            #        h[tttt-1] = len([ i for i in mind if bin_r[tttt-1] <= i < bin_r[tttt] ])
            #    else:
            #        h[tttt-1] = len([ i for i in mind if bin_r[tttt-1] <= i <= bin_r[tttt-1] ])
            #### DEBUG
            #### DEBUG 2
            #h = array([0]*( maxbin ))
            #for r in mind:
            #    bn = int(ceil(r/dr))
            #    if bn < nbins:
            #        h[bn-1] += 1
            #### DEBUG 2
            # Make a running total of the binned data
            if first_add:
                hist1d = h / sum(h)
                first_add = False
            else:
                hist1d += h / sum(h)
            ###########################################################
        # Divide the histogram by the number of timesteps to get the
        # average.  For the COM atoms, normalize to the number of 
        # timesteps that the (de)protonated species exists, rather
        # than the total.
        if ( at1 == '-2' ):
            hist[p] = hist1d / ( ntimesteps - len(empty_prot) )
        elif ( at1 == '-3' ):
            hist[p] = hist1d / ( ntimesteps - len(empty_deprot) )
        elif ( at2 in hydronium ):
            hist[p] = hist1d / ( ntimesteps - len(empty_deprot) )
        else:
            hist[p] = hist1d / ntimesteps 
        timing( tmp, datetime.now(), p + ':' )

    # Get rid of coords dictionary since it isn't needed anymore
    coords = 0

    # For symmetric atom pairs (e.g. OW-OW or HW-HW), we need to zero
    # the first element of the RDF.  This is a consequence of 
    # calculating distances between all atom pairs, including the atom
    # with itself.  Not doing this will result in a peak at 0.
    if do_water:
        hist['OW-OW'][0] = 0.0
        hist['HW-HW'][0] = 0.0

    # Print out the distance calculation timing 
    timing( distTime, datetime.now(), 'Distance calculation time:' )

    ####################
    # Normalize the data
    ####################

    # Timer for normalizing data 
    normTime = datetime.now()

    # RDF is built using the equation:
    #
    # g_{ab}(r) = dn(r) / normfac
    # normfac = rho_{a} * V_{shell} * N_{ts} * ( N - 1 ) / 3
    # rho_{a} = N_{a} / V_{total}
    # V_{shell} = (4/3) * pi * ( r_{i}^3 - r_{i-1}^3 )
    # 
    # Here:
    # dn(r) is the number of particles in a particular bin
    # rho_{a} is the number density of the system
    # N_{a} is the total number of type 'a' particles in the system
    # V_{total} is the total volume of the simulation box
    # V_{shell} is the volume of a spherical shell
    # N_{ts} is the number of timesteps in the simulation (this was done above)

    # Broadcast the numpy arrays to quickly find shell volumes
    shvol = (4.0/3.0) * pi * ( power(bin_r[1:],3) - power(bin_r[:-1],3) )

    for key in hist.keys():
        # Number density of the first atom type in the RDF.
        #at1 = key.split('-')[0]
        #rho = nattype[at1] / vtotal
        rho = 1.0 / vtotal
        # Normalization factor
        #normfac = rho * (natoms-1)/3.0
        normfac = rho
        for i in range(len(shvol)):
            # hist is now g(r)
            hist[key][i] = hist[key][i] / ( normfac * shvol[i] )
            # Store the normalization factor
            norm[key].append( normfac * shvol[i] )

    # Print out the normalization timing 
    timing( normTime, datetime.now(), 'Normalization time:' )

    #################
    # Output the data
    #################

    # Width of the bins (dr)
    dr = ( maxdist - mindist ) / ( nbins - 1 )

    # Calculate average bin position (midpoints of bins)
    avg_r = [ bin_r[i-1] + dr/2 for i in range(1, len(bin_r)) ] 

    # Output a file for each type of RDF calculated.
    for key in hist.keys():
        # Filename is based on the type of RDF
        fname = fprefix + 'RDF_' + key + '.dat'
        f = open(fname, 'w')
        # Format for output
        fmt = '{0:10.6f}    {1:12.6f}    {2:12.6f}'
        # Print data and header.
        print('# RDF file for ' + key, file=f)
        print( '# r (Angstrom)    g(r)         Normalization factor', 
               file=f ) 
        for i in range(len(avg_r)):
            print( fmt.format( avg_r[i], hist[key][i], norm[key][i] ), 
                   file=f )
        f.close()

    # Print out the final timing
    timing( startTime, datetime.now(), 'Total time:' )

def timing( start, current, tstring ):
    '''Output the current timing information.'''
    Time = current - start
    hour_min_sec = str(Time).split(':')
    hour_min_sec[0] = float(hour_min_sec[0])
    hour_min_sec[1] = float(hour_min_sec[1])
    hour_min_sec[2] = float(hour_min_sec[2])
    fmt = '{0:>26} {1:2.0f} {2} {3:2.0f} {4} {5:5.2f} {6}'
    print( fmt.format( tstring,
                       hour_min_sec[0], "hours,",
                       hour_min_sec[1], "minutes, and",
                       hour_min_sec[2], "seconds." ) )

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
